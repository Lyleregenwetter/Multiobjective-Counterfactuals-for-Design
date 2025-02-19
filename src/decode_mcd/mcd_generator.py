import os
from numbers import Real

import dill
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.evaluator import Evaluator
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling
from pymoo.core.population import Population
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.util.display.multi import MultiObjectiveOutput

from decode_mcd.mcd_problem import McdProblem, _MCD_BASE_OBJECTIVES, _PROXIMITY_INDEX, \
    _SPARSITY_INDEX, _MANPROX_INDEX
from decode_mcd_private import calculate_dtai as calculate_dtai, DPPsampling as DPPsampling
from decode_mcd_private.efficient_mixed_duplicate_elimination import EfficientMixedVariableDuplicateElimination
from decode_mcd_private.stats_methods import mixed_gower
from decode_mcd_private.validation_utils import validate

_DEFAULT_BETA = 4


# noinspection PyProtectedMember
class _RevertToQueryRepair(Repair):
    def __init__(self, alpha=0.2, beta=0.5, *args, **kwargs):
        self.alpha = alpha
        self.beta = beta
        super().__init__(*args, **kwargs)

    def _do(self, problem: McdProblem, Z, **kwargs):
        # noinspection PyProtectedMember
        revertible_variables = problem._revertible_variables
        original_query = problem._x_query[problem._features_to_vary]
        query_revertible = original_query[revertible_variables]
        Z_df = pd.DataFrame.from_records(Z)
        Z_revertible = Z_df[revertible_variables]
        Z_revertible = self._revert_subset(Z_revertible.values, query_revertible.values)
        Z_df[revertible_variables] = Z_revertible
        return Z_df.to_dict("records")

    def _revert_subset(self, Z_revertible, query_revertible):
        elementwise_prob = np.random.beta(self.alpha, self.beta)
        mask = np.random.binomial(size=np.shape(Z_revertible), n=1, p=elementwise_prob).astype(bool)
        Z_revertible[mask] = np.repeat(query_revertible, Z_revertible.shape[0], axis=0)[mask]
        return Z_revertible


class _AllOffspringCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["offspring"] = []

    def notify(self, algorithm):
        self.data["offspring"].append(algorithm.off)


# noinspection PyProtectedMember,PyMethodMayBeStatic
class McdGenerator:  # For calling the optimization and sampling counterfactuals
    def __init__(self,
                 mcd_problem: McdProblem,
                 pop_size: int,
                 initialize_from_dataset: bool = True,
                 verbose: bool = True):
        """An evolutionary-algorithm based counterfactuals generator.


        The generation and sampling steps are decoupled to allow users to vary sampling parameters without having to regenerate counterfactuals."""
        self._all_cf_y, self._all_cf_x, self._agg_scores, self._label_scores, \
            self._seed, self._res, self._algorithm, self._dataset_pop = (None for _ in range(8))
        self._problem = mcd_problem
        self._pop_size = pop_size
        self._initialize_from_dataset = initialize_from_dataset
        self._verbose = verbose
        self._validate_fields()

    def generate(self, n_generations: int, seed: int = None):  # Run the GA
        self._seed = self._get_or_default(seed, np.random.randint(1_000_000))

        self._validate_positive_int(n_generations, "n_generations")
        self._validate_positive_int(self._seed, "seed")

        self._setup_algorithm()
        previous_train_steps = self._get_or_default(self._algorithm.n_iter, 0)
        self._train_algorithm_if(n_generations, previous_train_steps)

    def save(self, filepath):
        self._verbose_log(f"Saving GA to {filepath}")
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        with open(f"{filepath}/checkpoint", "wb") as f:
            dill.dump(self._algorithm, f)

    def load(self, filepath):
        self._verbose_log(f"Loading GA from {filepath}")
        with open(f"{filepath}/checkpoint", "rb") as f:
            self._algorithm = dill.load(f)
            self._problem = self._algorithm.problem

    def sample_with_dtai(self, num_samples: int, manifold_proximity_weight: float = 0.5, sparsity_weight: float = 0.3, proximity_weight: float = 1,
                         diversity_weight: float = 0.3, dtai_target: np.ndarray = None,
                         dtai_alpha: np.ndarray = None, dtai_beta: np.ndarray = None,
                         include_dataset=True, max_dpp=None):  # Query from pareto front
        self._validate_sampling_parameters(num_samples, manifold_proximity_weight, sparsity_weight, proximity_weight, diversity_weight)
        dtai_target, dtai_alpha, dtai_beta = self._get_or_default_dtai(dtai_target, dtai_alpha, dtai_beta)
        self._validate_dtai_parameters(dtai_target, dtai_alpha, dtai_beta)

        all_cf_x, all_cf_y = self._initialize_and_filter_all_cfs(include_dataset)

        if len(all_cf_x) < num_samples:
            self._log_results_found(all_cf_x, all_cf_y)
            return self._build_res_df(all_cf_x)

        self._verbose_log("Scoring all counterfactual candidates!")

        agg_scores = self._calculate_scores_with_dtai(all_cf_y, manifold_proximity_weight, sparsity_weight,
                                                      dtai_alpha, dtai_beta, dtai_target,
                                                      proximity_weight)

        return self._sample_based_on_scores(all_cf_x, num_samples, diversity_weight, max_dpp, agg_scores)

    def sample(self, num_samples: int, manifold_proximity_weight: float = 0.5, sparsity_weight: float = 0.3, proximity_weight: float = 1,
               diversity_weight: float = 0.3, bonus_objectives_weights: np.ndarray = None, include_dataset=True,
               max_dpp=None):
        self._validate_sampling_parameters(num_samples, manifold_proximity_weight, sparsity_weight, proximity_weight, diversity_weight)
        bonus_objectives_weights = self._get_or_default(bonus_objectives_weights,
                                                        np.ones(shape=(len(self._problem._bonus_objectives))))
        self._validate_y_weights(bonus_objectives_weights)

        all_cf_x, all_cf_y = self._initialize_and_filter_all_cfs(include_dataset)

        if len(all_cf_x) < num_samples:
            self._log_results_found(all_cf_x, all_cf_y)
            return self._build_res_df(all_cf_x)

        self._verbose_log("Scoring all counterfactual candidates!")

        agg_scores = self._calculate_scores_with_weights(all_cf_y, manifold_proximity_weight, sparsity_weight, proximity_weight,
                                                         bonus_objectives_weights)

        return self._sample_based_on_scores(all_cf_x, num_samples, diversity_weight, max_dpp, agg_scores)

    def _validate_sampling_parameters(self, num_samples, manifold_proximity_weight, sparsity_weight, proximity_weight, diversity_weight):
        assert self._res, "You must call generate before calling sample!"
        self._validate_positive_int(num_samples, "num_samples")
        self._validate_statistical_weights(manifold_proximity_weight, sparsity_weight, proximity_weight, diversity_weight)

    def _setup_algorithm(self):  # First time algorithm setup
        if self._algorithm is None:  # Runs if algorithm is not yet initialized
            x = self._problem._x_query.loc[:, self._problem._features_to_vary].to_dict(
                "records")
            query_pop = Population.new("X", x)
            Evaluator().eval(self._problem,
                             query_pop)  # TODO: Concatenate before evaluating the query to save one call to evaluate?
            pop = self._initialize_population(query_pop)
            self._algorithm = self._build_algorithm(pop)

    def _build_algorithm(self, population):
        algorithm = NSGA2(pop_size=self._pop_size, sampling=population,
                          mating=MixedVariableMating(eliminate_duplicates=EfficientMixedVariableDuplicateElimination(),
                                                     repair=_RevertToQueryRepair()),
                          eliminate_duplicates=NoDuplicateElimination(),
                          # no duplicate elimination when initializing from dataset
                          callback=_AllOffspringCallback(),
                          output=MultiObjectiveOutput(),  # this is necessary because this object is mutable
                          save_history=False)
        algorithm.eliminate_duplicates = EfficientMixedVariableDuplicateElimination()  # set efficient duplicate elimination
        return algorithm

    def _initialize_population(self, query_pop):
        if self._initialize_from_dataset:
            self._generate_dataset_pop()
            self._verbose_log(f"Initial population initialized from dataset of {len(self._dataset_pop)} samples!")
            pop = Population.merge(self._dataset_pop, query_pop)
        else:
            mvs = MixedVariableSampling()
            pop = mvs(self._problem, self._pop_size - 1)
            self._verbose_log("Initial population randomly initialized!")
            pop = Population.merge(pop, query_pop)
        return pop

    def _verbose_log(self, log_message):
        if self._verbose:
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
        self._algorithm.termination = MaximumGenerationTermination(n_generations)
        self._res = minimize(self._problem, self._algorithm,
                             seed=self._seed,
                             copy_algorithm=False,
                             verbose=self._verbose)

    def _initialize_and_filter_all_cfs(self, include_dataset):
        all_cfs = self._initialize_all_cfs(include_dataset)
        all_cf_x, all_cf_y = self._filter_by_validity(all_cfs)
        self._all_cf_x = all_cf_x
        self._all_cf_y = all_cf_y
        return all_cf_x, all_cf_y

    def _sample_based_on_scores(self, all_cf_x, num_samples, diversity_weight, max_dpp, agg_scores):
        agg_scores = (agg_scores-np.min(agg_scores))/(np.max(agg_scores)-np.min(agg_scores)) #scale the scores from 0 to 1
        if num_samples == 1:
            best_idx = np.argmin(agg_scores)
            result = self._build_res_df(all_cf_x[best_idx:best_idx + 1, :])
            return self._check_for_original_query(result)
        else:
            if diversity_weight == 0:
                idx = np.argpartition(agg_scores, num_samples)[:num_samples]
                result = self._build_res_df(all_cf_x[idx, :])
                return self._check_for_original_query(result)
            else:
                if diversity_weight < 0.1:
                    self._log(
                        """Warning: Very small diversity can cause numerical instability. 
                        We recommend keeping diversity above 0.1 or setting diversity to 0""")
                if max_dpp and len(agg_scores) > max_dpp:
                    index = np.argpartition(agg_scores, max_dpp)[:max_dpp]
                else:
                    if len(agg_scores) > 25000:
                        self._log(
                        """Warning: Sampling a diverse set from over 25k possible counterfactuals. 
                        If this operation hangs or takes too much memory, 
                        consider setting max_dpp to limit the subset of solutions to sample from.""")
                    index = range(len(agg_scores))
                samples_index = self._diverse_sample(all_cf_x[index], agg_scores[index], num_samples, diversity_weight)
                result = self._build_res_df(all_cf_x[samples_index, :])
                return self._check_for_original_query(result)

    def _calculate_scores_with_dtai(self, all_cf_y, manifold_proximity_weight, sparsity_weight, dtai_alpha, dtai_beta, dtai_target,
                                    proximity_weight):
        dtai_alpha = np.array([dtai_alpha])
        dtai_beta = np.array([dtai_beta])
        dtai_target = np.array([dtai_target])
        dtai_scores = self._calculate_dtai(all_cf_y, dtai_alpha, dtai_beta, dtai_target)
        return self._calculate_statistical_scores(all_cf_y, manifold_proximity_weight, sparsity_weight, proximity_weight, dtai_scores)

    def _calculate_scores_with_weights(self, all_cf_y, manifold_proximity_weight, sparsity_weight, proximity_weight,
                                       bonus_objectives_weights):
        bonus_objectives_weights = np.array([bonus_objectives_weights])
        weighted_scores = np.sum(all_cf_y[:, :-_MCD_BASE_OBJECTIVES] * bonus_objectives_weights, axis=1)
        return self._calculate_statistical_scores(all_cf_y, manifold_proximity_weight, sparsity_weight, proximity_weight, weighted_scores)

    def _calculate_statistical_scores(self, all_cf_y, manifold_proximity_weight, sparsity_weight, proximity_weight, label_scores):
        cf_quality = all_cf_y[:, _PROXIMITY_INDEX] * proximity_weight + \
                     all_cf_y[:, _SPARSITY_INDEX] * sparsity_weight + \
                     all_cf_y[:, _MANPROX_INDEX] * manifold_proximity_weight
        agg_scores = label_scores + cf_quality
        # For quick debugging
        self._label_scores = label_scores
        self._agg_scores = agg_scores
        return agg_scores

    def _check_for_original_query(self, result):
        # Check if the initial query is in the final returned set
        # noinspection PyUnresolvedReferences
        if (result == self._problem._x_query.values).all(1).any():
            self._log("Initial Query is valid and included in the top counterfactuals identified")
        return result

    def _calculate_dtai(self, all_cf_y, dtai_alpha, dtai_beta, dtai_target):
        return calculate_dtai.calculateDTAI(all_cf_y[:, :-_MCD_BASE_OBJECTIVES], "minimize", dtai_target, dtai_alpha,
                                            dtai_beta)

    def _initialize_all_cfs(self, include_dataset):
        self._verbose_log("Collecting all counterfactual candidates!")
        if include_dataset:
            self._generate_dataset_pop()
            all_cfs = self._dataset_pop
        else:
            all_cfs = Population()
        for offspring in self._res.algorithm.callback.data["offspring"]:
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

        valid = np.all(1 - np.sign(all_cf_v), axis=1)
        return all_cf_x[valid], all_cf_y[valid]

    def _min2max(self, x):  # Converts minimization objective to maximization, assumes rough scale~ 1
        return 1-x

    def _build_res_df(self, x):
        self._verbose_log("Done! Returning CFs")
        # noinspection PyProtectedMember
        return pd.DataFrame(self._problem._build_full_df(x),
                            columns=self._problem._valid_features_dataset.columns)

    def _generate_dataset_pop(self):
        # TODO remove any that are out of range or that change features that are supposed to be fixed
        if self._dataset_pop is None:  # Evaluate Pop if not done already
            x = self._problem._valid_features_dataset
            x = x.loc[:, self._problem._features_to_vary].to_dict("records")
            pop = Population.new("X", x)
            Evaluator().eval(self._problem, pop, datasetflag=True)
            self._dataset_pop = pop
            self._verbose_log(f"{len(pop)} dataset entries found matching problem parameters")

    # noinspection PyProtectedMember
    def _diverse_sample(self, x, y, num_samples, diversity_weight):
        self._verbose_log("Calculating diversity matrix!")
        y = np.power(self._min2max(y), 1 / diversity_weight)
        x_df = self._problem._build_full_df(x)
        matrix = mixed_gower(x_df, x_df, self._problem._ranges.values, self._problem._build_feature_types())
        weighted_matrix = np.einsum('ij,i,j->ij', matrix, y, y)
        self._verbose_log("Sampling diverse set of counterfactual candidates!")
        samples_index = DPPsampling.pure_greedy(weighted_matrix, num_samples)
        return samples_index

    def _get_near_psd(self, A):
        C = (A + A.T) / 2
        eigval, eigvec = np.linalg.eig(C)
        eigval[eigval < 0] = 0

        return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

    def _validate_fields(self):
        validate(isinstance(self._problem, McdProblem), "problem must be an instance "
                                                                   "of decode_mcd.MultiObjectiveProblem")
        self._validate_positive_int(self._pop_size, "pop_size")

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
        self._validate_bonus_objective_scoring_parameter(y_weights, "y_weights")

    def _validate_bonus_objective_scoring_parameter(self, parameter: np.ndarray, parameter_name):
        validate(isinstance(parameter, np.ndarray), f"{parameter_name} must be a numpy array")
        n_bonus = len(self._problem._bonus_objectives)
        expected_shape = (n_bonus,)
        exception_message = self._get_exception_message(expected_shape, n_bonus, parameter_name)
        validate(parameter.shape == expected_shape, exception_message)

    def _get_exception_message(self, expected_shape, n_bonus, parameter_name):
        if n_bonus == 0:
            return f"No bonus objectives are set - {parameter_name} must be left empty!"
        else:
            return f"{parameter_name} must have shape {expected_shape} " \
                   f"to match the number of bonus objectives"

    def _validate_dtai_parameters(self, dtai_target, dtai_alpha, dtai_beta):
        self._validate_bonus_objective_scoring_parameter(dtai_target, "dtai_target")
        self._validate_bonus_objective_scoring_parameter(dtai_alpha, "dtai_alpha")
        self._validate_bonus_objective_scoring_parameter(dtai_beta, "dtai_beta")

    def _get_or_default_dtai(self, dtai_target, dtai_alpha, dtai_beta):
        dtai_target = self._get_or_default(dtai_target,
                                           np.ones(shape=(len(self._problem._bonus_objectives))))
        dtai_alpha = self._get_or_default(dtai_alpha, np.ones_like(dtai_target))
        dtai_beta = self._get_or_default(dtai_beta, np.ones_like(dtai_target) * _DEFAULT_BETA)
        return dtai_target, dtai_alpha, dtai_beta
