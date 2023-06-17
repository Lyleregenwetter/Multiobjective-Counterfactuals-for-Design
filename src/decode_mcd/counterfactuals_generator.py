import os

import dill
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.evaluator import Evaluator
from pymoo.core.mixed import MixedVariableMating, MixedVariableDuplicateElimination, MixedVariableSampling
from pymoo.core.population import Population
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination

from decode_mcd.multi_objective_problem import MultiObjectiveProblem, MCD_BASE_OBJECTIVES
from private import calculate_dtai as calculate_dtai, DPPsampling as DPPsampling
from private.stats_methods import mixed_gower

DEFAULT_BETA = 4


class RevertToQueryRepair(Repair):
    def __init__(self, rep_prob=0.2, elementwise_prob=0.3, *args, **kwargs):
        self.rep_prob = rep_prob
        self.elementwise_prob = elementwise_prob
        super().__init__(*args, **kwargs)

    def _do(self, problem: MultiObjectiveProblem, Z, **kwargs):
        qxs = problem.data_package.query_x.loc[:, problem.data_package.features_to_vary]
        Z_pd = pd.DataFrame.from_records(Z)
        Z_np = Z_pd.values
        mask = np.random.binomial(size=np.shape(Z_np), n=1, p=self.elementwise_prob)
        mask = mask * np.random.binomial(size=(np.shape(Z_np)[0], 1), n=1, p=self.rep_prob)
        Z_np = qxs.values * mask + Z_np * (1 - mask)
        Z = pd.DataFrame(Z_np, columns=Z_pd.columns)
        return Z.to_dict("records")


class AllOffspringCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["offspring"] = []

    def notify(self, algorithm):
        self.data["offspring"].append(algorithm.off)


class CounterfactualsGenerator:  # For calling the optimization and sampling counterfactuals
    def __init__(self, problem: MultiObjectiveProblem,
                 pop_size: int,
                 initialize_from_dataset: bool = True,
                 verbose: bool = True):
        self.all_cf_y, self.all_cf_x, self.agg_scores, self.dtai_scores, \
            self.seed, self.res, self.algorithm, self.dataset_pop = (None for _ in range(8))
        self.problem = problem
        self.pop_size = pop_size
        self.initialize_from_dataset = initialize_from_dataset
        self.verbose = verbose

    def initialize(self):  # First time algorithm setup
        if self.algorithm is None:  # Runs if algorithm is not yet initialized
            x = self.problem.data_package.query_x.loc[:, self.problem.data_package.features_to_vary].to_dict("records")
            query_pop = Population.new("X", x)
            Evaluator().eval(self.problem,
                             query_pop)  # TODO: Concatenate before evaluating the query to save one call to evaluate?
            pop = self._initialize_population(query_pop)
            self.algorithm = self._build_algorithm(pop)

    def _build_algorithm(self, population):
        return NSGA2(pop_size=self.pop_size, sampling=population,
                     mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination(),
                                                repair=RevertToQueryRepair()),
                     eliminate_duplicates=MixedVariableDuplicateElimination(),
                     callback=AllOffspringCallback(),
                     save_history=False)

    def _initialize_population(self, query_pop):
        if self.initialize_from_dataset:
            self.generate_dataset_pop()
            self._verbose_log(f"Initial population initialized from dataset of {len(self.dataset_pop)} samples!")
            pop = Population.merge(self.dataset_pop, query_pop)
        else:
            mvs = MixedVariableSampling()
            pop = mvs(self.problem, self.pop_size - 1)
            self._verbose_log("Initial population randomly initialized!")
            pop = Population.merge(pop, query_pop)
        return pop

    def _verbose_log(self, log_message):
        if self.verbose:
            print(log_message)

    def _get_or_default(self, value, default_value):
        if value is None:
            return default_value
        return value

    def generate(self, n_generations, seed=None):  # Run the GA

        self.seed = self._get_or_default(seed, np.random.randint(1_000_000))

        self.initialize()

        previous_train_steps = self._get_or_default(self.algorithm.n_iter, 0)

        if n_generations >= previous_train_steps:
            self._verbose_log(f"Training GA from {previous_train_steps} to {n_generations} generations!")
            self.algorithm.termination = MaximumGenerationTermination(n_generations)
            self.res = minimize(self.problem, self.algorithm,
                                seed=self.seed,
                                copy_algorithm=False,
                                verbose=self.verbose)
        else:
            print(f"GA has already trained for {previous_train_steps} generations.")

    def save(self, filepath):
        self._verbose_log(f"Saving GA to {filepath}")
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        with open(f"{filepath}/checkpoint", "wb") as f:
            dill.dump(self.algorithm, f)

    def load(self, filepath):
        self._verbose_log(f"Loading GA from {filepath}")
        with open(f"{filepath}/checkpoint", "rb") as f:
            self.algorithm = dill.load(f)
            self.problem = self.algorithm.problem

    def sample(self, num_samples: int, avg_gower_weight, cfc_weight, gower_weight, diversity_weight, dtai_target,
               dtai_alpha=None, dtai_beta=None, include_dataset=True, num_dpp=1000):  # Query from pareto front
        assert self.res, "You must call optimize before calling generate!"
        assert num_samples > 0, "You must sample at least 1 counterfactual!"

        all_cfs = self._initialize_all_cfs(include_dataset)

        all_cf_x, all_cf_y = self._filter_by_validity(all_cfs)
        self.all_cf_x = all_cf_x
        self.all_cf_y = all_cf_y

        if len(all_cf_x) < num_samples:  # bug
            self._log_results_found(all_cf_x, all_cf_y)
            return self.build_res_df(all_cf_x)

        self._verbose_log("Scoring all counterfactual candidates!")

        dtai_scores = self._calculate_dtai(all_cf_y, dtai_alpha, dtai_beta, dtai_target)
        cf_quality = all_cf_y[:, -3] * gower_weight + all_cf_y[:, -2] * cfc_weight + all_cf_y[:, -1] * avg_gower_weight
        agg_scores = 1 - dtai_scores + cf_quality

        # For quick debugging

        self.dtai_scores = dtai_scores
        self.agg_scores = agg_scores

        if num_samples == 1:
            best_idx = np.argmin(agg_scores)
            result = self.build_res_df(all_cf_x[best_idx:best_idx + 1, :])
            return self.final_check(result)
        else:
            if diversity_weight == 0:
                idx = np.argpartition(agg_scores, num_samples)
                result = self.build_res_df(all_cf_x[idx, :])
                return self.final_check(result)
            else:
                if diversity_weight < 0.1:
                    print(
                        """Warning: Very small diversity can cause numerical instability. 
                        We recommend keeping diversity above 0.1 or setting diversity to 0""")
                if len(agg_scores) > num_dpp:
                    index = np.argpartition(agg_scores, -num_dpp)[-num_dpp:]
                else:
                    index = range(len(agg_scores))
                samples_index = self.diverse_sample(all_cf_x[index], agg_scores[index], num_samples, diversity_weight)
                result = self.build_res_df(all_cf_x[samples_index, :])
                return self.final_check(result)

    def final_check(self, result):
        print(self.problem.data_package.query_x)
        print(result)
        # Check if the initial query is in the final returned set
        if (result == self.problem.data_package.query_x.values).all(1).any():
            print("Initial Query is valid and included in the top counterfactuals identified")
        return result

    def _calculate_dtai(self, all_cf_y, dtai_alpha, dtai_beta, dtai_target):
        if dtai_alpha is None:
            dtai_alpha = np.ones_like(dtai_target)
        if dtai_beta is None:
            dtai_beta = np.ones_like(dtai_target) * DEFAULT_BETA
        return calculate_dtai.calculateDTAI(all_cf_y[:, :-MCD_BASE_OBJECTIVES], "minimize", dtai_target, dtai_alpha,
                                            dtai_beta)

    def _initialize_all_cfs(self, include_dataset):
        self._verbose_log("Collecting all counterfactual candidates!")
        if include_dataset:
            self.generate_dataset_pop()
            all_cfs = self.dataset_pop
        else:
            all_cfs = Population()
        for offspring in self.res.algorithm.callback.data["offspring"]:
            all_cfs = Population.merge(all_cfs, offspring)
        return all_cfs

    def _log_results_found(self, all_cf_x, all_cf_y):
        if len(all_cf_x) == 0:
            print(f"No valid counterfactuals! Returning empty dataframe.")
        else:
            number_found = len(all_cf_y)
            print(f"Only found {number_found} valid counterfactuals! Returning all {number_found}.")

    def _filter_by_validity(self, all_cfs):
        all_cf_y = all_cfs.get("F")
        all_cf_v = all_cfs.get("G")
        all_cf_x = all_cfs.get("X")
        all_cf_x = pd.DataFrame.from_records(all_cf_x).values

        valid = np.all(1 - all_cf_v, axis=1)
        return all_cf_x[valid], all_cf_y[valid]

    def min2max(self, x, eps=1e-7):  # Converts minimization objective to maximization, assumes rough scale~ 1
        return np.divide(np.mean(x), x + eps)

    def build_res_df(self, x):
        self._verbose_log("Done! Returning CFs")
        # noinspection PyProtectedMember
        return pd.DataFrame(self.problem._build_full_df(x), columns=self.problem.data_package.features_dataset.columns)

    def generate_dataset_pop(self):
        # TODO remove any that are out of range or that change features that are supposed to be fixed
        if self.dataset_pop is None:  # Evaluate Pop if not done already
            x = self.problem.data_package.features_dataset
            x = x.loc[:, self.problem.data_package.features_to_vary].to_dict("records")
            pop = Population.new("X", x)
            Evaluator().eval(self.problem, pop, datasetflag=True)
            self.dataset_pop = pop
            self._verbose_log(f"{len(pop)} dataset entries found matching problem parameters")

    # noinspection PyProtectedMember
    def diverse_sample(self, x, y, num_samples, diversity_weight, eps=1e-7):
        self._verbose_log("Calculating diversity matrix!")
        y = np.power(self.min2max(y), 1 / diversity_weight)
        x_df = self.problem._build_full_df(x)
        matrix = mixed_gower(x_df, x_df, self.problem.ranges.values, self.problem._build_gower_types())
        weighted_matrix = np.einsum('ij,i,j->ij', matrix, y, y)
        self._verbose_log("Sampling diverse set of counterfactual candidates!")
        samples_index = DPPsampling.kDPPGreedySample(weighted_matrix, num_samples)
        return samples_index