import os
from numbers import Real

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
from pymoo.util.display.multi import MultiObjectiveOutput

from decode_mcd.multi_objective_problem import MultiObjectiveProblem, _MCD_BASE_OBJECTIVES, _GOWER_INDEX, \
    _CHANGED_FEATURE_INDEX, _AVG_GOWER_INDEX
from decode_mcd_private import calculate_dtai as calculate_dtai, DPPsampling as DPPsampling
from decode_mcd_private.stats_methods import mixed_gower
from decode_mcd_private.validation_utils import validate

_DEFAULT_BETA = 4


class RevertToQueryRepair(Repair):
    def __init__(self, rep_prob=0.2, elementwise_prob=0.3, *args, **kwargs):
        self.rep_prob = rep_prob
        self.elementwise_prob = elementwise_prob
        super().__init__(*args, **kwargs)

    def _do(self, problem: MultiObjectiveProblem, Z, **kwargs):
        # noinspection PyProtectedMember
        revertible_indexes = problem._revertible_indexes
        qxs = problem.data_package.query_x.values[:, revertible_indexes]
        full_Z_dataframe = pd.DataFrame.from_records(Z)
        full_Z_np = full_Z_dataframe.values
        reverted_subset = self._revert_subset(full_Z_np, qxs, revertible_indexes)
        full_Z_np[:, revertible_indexes] = reverted_subset
        Z = pd.DataFrame(full_Z_np, columns=full_Z_dataframe.columns)
        return Z.to_dict("records")

    def _revert_subset(self, full_Z_np, qxs, revertible_indexes):
        revertible_subset = full_Z_np[:, revertible_indexes]
        mask = np.random.binomial(size=np.shape(revertible_subset), n=1, p=self.elementwise_prob)
        mask = mask * np.random.binomial(size=(np.shape(revertible_subset)[0], 1), n=1, p=self.rep_prob)
        revertible_subset = qxs * mask + revertible_subset * (1 - mask)
        return revertible_subset


class AllOffspringCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["offspring"] = []

    def notify(self, algorithm):
        self.data["offspring"].append(algorithm.off)


class CounterfactualsGenerator:  # For calling the optimization and sampling counterfactuals
    def __init__(self,
                 problem: MultiObjectiveProblem,
                 pop_size: int,
                 initialize_from_dataset: bool = True,
                 verbose: bool = True):
        self.all_cf_y, self.all_cf_x, self.agg_scores, self.label_scores, \
            self.seed, self.res, self.algorithm, self.dataset_pop = (None for _ in range(8))
        self.problem = problem
        self.pop_size = pop_size
        self.initialize_from_dataset = initialize_from_dataset
        self.verbose = verbose
        self._validate_fields()

    def generate(self, n_generations: int, seed: int = None):  # Run the GA
        self.seed = self._get_or_default(seed, np.random.randint(1_000_000))

        self._validate_positive_int(n_generations, "n_generations")
        self._validate_positive_int(self.seed, "seed")

        self._setup_algorithm()
        previous_train_steps = self._get_or_default(self.algorithm.n_iter, 0)
        self._train_algorithm_if(n_generations, previous_train_steps)

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

    def sample_with_dtai(self, num_samples: int, avg_gower_weight, cfc_weight, gower_weight,
                         diversity_weight, dtai_target, dtai_alpha=None, dtai_beta=None,
                         include_dataset=True, num_dpp=1000):  # Query from pareto front
        self._validate_sampling_parameters(num_samples, avg_gower_weight, cfc_weight, gower_weight, diversity_weight)
        self._validate_dtai_parameters(dtai_target, dtai_alpha, dtai_beta)

        all_cf_x, all_cf_y = self._initialize_and_filter_all_cfs(include_dataset)

        if len(all_cf_x) < num_samples:
            self._log_results_found(all_cf_x, all_cf_y)
            return self._build_res_df(all_cf_x)

        self._verbose_log("Scoring all counterfactual candidates!")

        agg_scores = self._calculate_scores_with_dtai(all_cf_y, avg_gower_weight, cfc_weight,
                                                      dtai_alpha, dtai_beta, dtai_target,
                                                      gower_weight)

        return self._sample_based_on_scores(all_cf_x, num_samples, diversity_weight, num_dpp, agg_scores)

    def sample_with_weights(self, num_samples: int, avg_gower_weight, cfc_weight, gower_weight,
                            diversity_weight, bonus_objectives_weights, include_dataset=True, num_dpp=1000):
        self._validate_sampling_parameters(num_samples, avg_gower_weight, cfc_weight, gower_weight, diversity_weight)
        self._validate_y_weights(bonus_objectives_weights)

        all_cf_x, all_cf_y = self._initialize_and_filter_all_cfs(include_dataset)

        if len(all_cf_x) < num_samples:
            self._log_results_found(all_cf_x, all_cf_y)
            return self._build_res_df(all_cf_x)

        self._verbose_log("Scoring all counterfactual candidates!")

        agg_scores = self._calculate_scores_with_weights(all_cf_y, avg_gower_weight, cfc_weight, gower_weight,
                                                         bonus_objectives_weights)

        return self._sample_based_on_scores(all_cf_x, num_samples, diversity_weight, num_dpp, agg_scores)

    def _validate_sampling_parameters(self, num_samples, avg_gower_weight, cfc_weight, gower_weight, diversity_weight):
        assert self.res, "You must call generate before calling sample!"
        self._validate_positive_int(num_samples, "num_samples")
        self._validate_statistical_weights(avg_gower_weight, cfc_weight, gower_weight, diversity_weight)

    def _setup_algorithm(self):  # First time algorithm setup
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
                     output=MultiObjectiveOutput(),  # this is necessary because this object is mutable
                     save_history=False)

    def _initialize_population(self, query_pop):
        if self.initialize_from_dataset:
            self._generate_dataset_pop()
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

    def _log(self, log_message):
        print(log_message)

    def _get_or_default(self, value, default_value):
        if value is None:
            return default_value
        return value

    def _train_algorithm_if(self, n_generations, previous_train_steps):
        if n_generations >= previous_train_steps:
            self._train_algorithm(n_generations, previous_train_steps)
        else:
            self._log(f"GA has already trained for {previous_train_steps} generations.")

    def _train_algorithm(self, n_generations, previous_train_steps):
        self._verbose_log(f"Training GA from {previous_train_steps} to {n_generations} generations!")
        self.algorithm.termination = MaximumGenerationTermination(n_generations)
        self.res = minimize(self.problem, self.algorithm,
                            seed=self.seed,
                            copy_algorithm=False,
                            verbose=self.verbose)

    def _initialize_and_filter_all_cfs(self, include_dataset):
        all_cfs = self._initialize_all_cfs(include_dataset)
        all_cf_x, all_cf_y = self._filter_by_validity(all_cfs)
        self.all_cf_x = all_cf_x
        self.all_cf_y = all_cf_y
        return all_cf_x, all_cf_y

    def _sample_based_on_scores(self, all_cf_x, num_samples, diversity_weight, num_dpp, agg_scores):
        if num_samples == 1:
            best_idx = np.argmin(agg_scores)
            result = self._build_res_df(all_cf_x[best_idx:best_idx + 1, :])
            return self._check_for_original_query(result)
        else:
            if diversity_weight == 0:
                idx = np.argpartition(agg_scores, num_samples)
                result = self._build_res_df(all_cf_x[idx, :])
                return self._check_for_original_query(result)
            else:
                if diversity_weight < 0.1:
                    self._log(
                        """Warning: Very small diversity can cause numerical instability. 
                        We recommend keeping diversity above 0.1 or setting diversity to 0""")
                if len(agg_scores) > num_dpp:
                    index = np.argpartition(agg_scores, -num_dpp)[-num_dpp:]
                else:
                    index = range(len(agg_scores))
                samples_index = self._diverse_sample(all_cf_x[index], agg_scores[index], num_samples, diversity_weight)
                result = self._build_res_df(all_cf_x[samples_index, :])
                return self._check_for_original_query(result)

    def _calculate_scores_with_dtai(self, all_cf_y, avg_gower_weight, cfc_weight, dtai_alpha, dtai_beta, dtai_target,
                                    gower_weight):
        dtai_scores = self._calculate_dtai(all_cf_y, dtai_alpha, dtai_beta, dtai_target)
        return self._calculate_statistical_scores(all_cf_y, avg_gower_weight, cfc_weight, gower_weight, dtai_scores)

    def _calculate_scores_with_weights(self, all_cf_y, avg_gower_weight, cfc_weight, gower_weight,
                                       bonus_objectives_weights):
        weighted_scores = np.sum(all_cf_y[:, :-_MCD_BASE_OBJECTIVES] * bonus_objectives_weights, axis=1)
        return self._calculate_statistical_scores(all_cf_y, avg_gower_weight, cfc_weight, gower_weight, weighted_scores)

    def _calculate_statistical_scores(self, all_cf_y, avg_gower_weight, cfc_weight, gower_weight, label_scores):
        cf_quality = all_cf_y[:, _GOWER_INDEX] * gower_weight + \
                     all_cf_y[:, _CHANGED_FEATURE_INDEX] * cfc_weight + \
                     all_cf_y[:, _AVG_GOWER_INDEX] * avg_gower_weight
        agg_scores = 1 - label_scores + cf_quality
        # For quick debugging
        self.label_scores = label_scores
        self.agg_scores = agg_scores
        return agg_scores

    def _check_for_original_query(self, result):
        # Check if the initial query is in the final returned set
        if (result == self.problem.data_package.query_x.values).all(1).any():
            self._log("Initial Query is valid and included in the top counterfactuals identified")
        return result

    def _calculate_dtai(self, all_cf_y, dtai_alpha, dtai_beta, dtai_target):
        dtai_alpha = self._get_or_default(dtai_alpha, np.ones_like(dtai_target))
        dtai_beta = self._get_or_default(dtai_beta, np.ones_like(dtai_target) * _DEFAULT_BETA)
        return calculate_dtai.calculateDTAI(all_cf_y[:, :-_MCD_BASE_OBJECTIVES], "minimize", dtai_target, dtai_alpha,
                                            dtai_beta)

    def _initialize_all_cfs(self, include_dataset):
        self._verbose_log("Collecting all counterfactual candidates!")
        if include_dataset:
            self._generate_dataset_pop()
            all_cfs = self.dataset_pop
        else:
            all_cfs = Population()
        for offspring in self.res.algorithm.callback.data["offspring"]:
            all_cfs = Population.merge(all_cfs, offspring)
        return all_cfs

    def _log_results_found(self, all_cf_x, all_cf_y):
        if len(all_cf_x) == 0:
            self._log(f"No valid counterfactuals! Returning empty dataframe.")
        else:
            number_found = len(all_cf_y)
            self._log(f"Only found {number_found} valid counterfactuals! Returning all {number_found}.")

    def _filter_by_validity(self, all_cfs):
        all_cf_y = all_cfs.get("F")
        all_cf_v = all_cfs.get("G")
        all_cf_x = all_cfs.get("X")
        all_cf_x = pd.DataFrame.from_records(all_cf_x).values

        valid = np.all(1 - all_cf_v, axis=1)
        return all_cf_x[valid], all_cf_y[valid]

    def _min2max(self, x, eps=1e-7):  # Converts minimization objective to maximization, assumes rough scale~ 1
        return np.divide(np.mean(x), x + eps)

    def _build_res_df(self, x):
        self._verbose_log("Done! Returning CFs")
        # noinspection PyProtectedMember
        return pd.DataFrame(self.problem._build_full_df(x), columns=self.problem.data_package.features_dataset.columns)

    def _generate_dataset_pop(self):
        # TODO remove any that are out of range or that change features that are supposed to be fixed
        if self.dataset_pop is None:  # Evaluate Pop if not done already
            x = self.problem.data_package.features_dataset
            x = x.loc[:, self.problem.data_package.features_to_vary].to_dict("records")
            pop = Population.new("X", x)
            Evaluator().eval(self.problem, pop, datasetflag=True)
            self.dataset_pop = pop
            self._verbose_log(f"{len(pop)} dataset entries found matching problem parameters")

    # noinspection PyProtectedMember
    def _diverse_sample(self, x, y, num_samples, diversity_weight, eps=1e-7):
        self._verbose_log("Calculating diversity matrix!")
        y = np.power(self._min2max(y), 1 / diversity_weight)
        x_df = self.problem._build_full_df(x)
        matrix = mixed_gower(x_df, x_df, self.problem.ranges.values, self.problem._build_gower_types())
        weighted_matrix = np.einsum('ij,i,j->ij', matrix, y, y)
        self._verbose_log("Sampling diverse set of counterfactual candidates!")
        samples_index = DPPsampling.kDPPGreedySample(weighted_matrix, num_samples)
        return samples_index

    def _validate_fields(self):
        validate(isinstance(self.problem, MultiObjectiveProblem), "problem must be an instance "
                                                                  "of decode_mcd.MultiObjectiveProblem")
        self._validate_positive_int(self.pop_size, "pop_size")

    def _validate_positive_int(self, value, param_name):
        validate(isinstance(value, int), f"{param_name} must be an integer")
        validate(value > 0, f"{param_name} must be a positive integer")

    def _validate_non_negative_float(self, value, param_name):
        validate(isinstance(value, Real), f"{param_name} must be a real number")
        validate(value >= 0, f"{param_name} must be >= 0")

    def _validate_statistical_weights(self, *weights):
        for weight in weights:
            self._validate_non_negative_float(weight, "weight")

    def _validate_y_weights(self, y_weights: np.ndarray):
        validate(isinstance(y_weights, np.ndarray), "y_weights must be a numpy array")
        expected_shape = (1, len(self.problem.data_package.bonus_objectives))
        validate(y_weights.shape == expected_shape,
                 f"y_weights must have shape {expected_shape}")

    def _validate_dtai_parameters(self, dtai_target, dtai_alpha, dtai_beta):
        pass
