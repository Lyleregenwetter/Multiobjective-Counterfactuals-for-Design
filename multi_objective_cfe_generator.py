import itertools
import os
import re
from typing import List

import dill
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.evaluator import Evaluator
from pymoo.core.mixed import MixedVariableSampling, MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.core.population import Population
# from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.variable import Real, Integer, Binary, Choice
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination

import DPPsampling as DPPsampling
import calculate_dtai as calculate_dtai
from classification_evaluator import ClassificationEvaluator
from data_package import DataPackage
from design_targets import DesignTargets
from stats_methods import mixed_gower, avg_gower_distance, changed_features_ratio, to_dataframe

# from main.evaluation.Predictor import Predictor

MEANING_OF_LIFE = 42


class MultiObjectiveCounterfactualsGenerator(Problem):
    def __init__(self,
                 data_package: DataPackage,
                 predictor: callable,
                 constraint_functions: list,
                 datatypes: list = None):
        self.data_package = data_package
        self.number_of_objectives = len(data_package.bonus_objectives) + 3
        self.x_dimension = len(self.data_package.features_dataset.columns)
        self.predictor = predictor
        self.constraint_functions = constraint_functions
        self.datatypes = datatypes
        super().__init__(vars=self._build_problem_var_dict(),
                         n_obj=self.number_of_objectives,
                         n_constr=len(constraint_functions) + self._count_y_constraints())
        self.ranges = self._build_ranges(self.data_package.features_dataset)
        self._avg_gower_sample_size = 1000
        self._avg_gower_sample_seed = MEANING_OF_LIFE
        self._set_valid_datasets_subset()  # Remove any invalid designs from the features dataset and predictions
        # dataset

    def set_average_gower_sampling_parameters(self, sample_size: int, sample_seed: int):
        self._validate(sample_size > 0, "Invalid sample size; must be greater than zero")
        self._validate(sample_seed > 0, "Invalid seed; must be greater than zero")
        self._avg_gower_sample_size = sample_size
        self._avg_gower_sample_seed = sample_seed

    def _count_y_constraints(self):
        return self.data_package.design_targets.count_constrained_labels()

    def _build_problem_var_dict(self):
        variables = {}
        for i in range(len(self.data_package.features_to_vary)):
            variables[self.data_package.features_to_vary[i]] = self.datatypes[i]
        return variables

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        # This flag will avoid passing the dataset through the predictor, when the y values are already known
        dataset_flag = kwargs.get("datasetflag", False)
        score, validity = self._calculate_evaluation_metrics(x, dataset_flag)
        out["F"] = score
        out["G"] = validity

    def _calculate_evaluation_metrics(self, x: np.ndarray, dataset_flag: bool):
        x_full = self._build_full_df(x)
        predictions = self._get_predictions(x_full, dataset_flag)

        scores = self._get_scores(x_full, predictions)
        validity = self._get_mixed_constraint_satisfaction(x_full, predictions, self.constraint_functions,
                                                           self.data_package.design_targets)
        return scores, validity

    def _get_scores(self, x: pd.DataFrame, predictions: pd.DataFrame):
        all_scores = np.zeros((len(x), self.number_of_objectives))
        gower_types = self._build_gower_types()
        all_scores[:, :-3] = predictions.loc[:, self.data_package.bonus_objectives]
        all_scores[:, -3] = mixed_gower(x, self.data_package.query_x, self.ranges.values, gower_types).T
        all_scores[:, -2] = changed_features_ratio(x, self.data_package.query_x, self.x_dimension)
        subset = self._get_features_sample()
        all_scores[:, -1] = avg_gower_distance(x, subset, self.ranges.values, gower_types)
        return all_scores

    def _get_features_sample(self):
        subset_size = min(self._avg_gower_sample_size, len(self.data_package.features_dataset))
        subset = self.data_package.features_dataset.sample(n=subset_size, axis=0,
                                                           random_state=self._avg_gower_sample_seed)
        return subset

    def _get_predictions(self, x_full: pd.DataFrame, dataset_flag: bool):
        if dataset_flag:
            return self.data_package.predictions_dataset.copy()
        return pd.DataFrame(self.predictor(x_full), columns=self.data_package.predictions_dataset.columns)

    def _build_gower_types(self):
        return {
            "r": tuple(self._get_features_by_type([Real, Integer])),
            "c": tuple(self._get_features_by_type([Choice, Binary]))
        }

    def _set_valid_datasets_subset(self):
        # Scans the features_dataset and returns the subset violating the variable categories and ranges,
        # as well as the changed feature specifications
        f_d = self.data_package.features_dataset
        p_d = self.data_package.predictions_dataset
        q = self.data_package.query_x
        reals_and_ints_idx = self._get_features_by_type([Real, Integer])  # pass in the pymoo built in variable types
        for index in reals_and_ints_idx:  # Filter out any that don't fall into an acceptable range
            p_d = p_d[(f_d.iloc[:, index] >= self.datatypes[index].bounds[0])]
            f_d = f_d[(f_d.iloc[:, index] >= self.datatypes[index].bounds[0])]
            p_d = p_d[(f_d.iloc[:, index] <= self.datatypes[index].bounds[1])]
            f_d = f_d[(f_d.iloc[:, index] <= self.datatypes[index].bounds[1])]
        categorical_idx = self._get_features_by_type([Choice])  # pass in the pymoo built in variable types
        for parameter in categorical_idx:  # Filter out any that don't fall into an acceptable category
            p_d = p_d[(f_d.iloc[:, parameter].isin(self.datatypes[parameter].options))]
            f_d = f_d[(f_d.iloc[:, parameter].isin(self.datatypes[parameter].options))]  # TODO: fix this bug...

        f2f = self.data_package.features_to_freeze
        if len(f2f) > 0:
            f_d_view = f_d[f2f]
            query_view = self.data_package.query_x[f2f]
            p_d = p_d[np.equal(f_d_view.values, query_view.values)]
            f_d = f_d[np.equal(f_d_view.values, query_view.values)]
        self.data_package.features_dataset = f_d
        self.data_package.predictions_dataset = p_d

    def _get_features_by_type(self, types: list) -> list:
        """Helper function to get a list of parameter indices of a particular datatype"""
        dts = self.datatypes
        matching_idxs = []
        for i in range(len(dts)):
            if type(dts[i]) in types:
                matching_idxs.append(i)
        return matching_idxs

    def _build_ranges(self, features_dataset: pd.DataFrame):
        # TODO: question this. Do we build ranges based on the
        #  features dataset or based on the limits provided by the user in datatypes?
        indices = self._get_features_by_type([Real, Integer])
        numeric_features = features_dataset.iloc[:, indices]
        return numeric_features.max() - numeric_features.min()

    def _get_mixed_constraint_satisfaction(self,
                                           x_full: pd.DataFrame,
                                           y: pd.DataFrame,
                                           x_constraint_functions: list,
                                           design_targets: DesignTargets):
        n_total_constraints = design_targets.count_constrained_labels()
        n_rows = x_full.shape[0]
        result = np.zeros(shape=(n_rows, n_total_constraints))

        self._append_x_constraint_satisfaction(result, x_full, x_constraint_functions, n_total_constraints)
        self._append_proba_satisfaction(result, y, design_targets)
        self._append_satisfaction(result, self._evaluate_regression_satisfaction, y, design_targets,
                                  design_targets.get_continuous_labels())
        self._append_satisfaction(result, self._evaluate_categorical_satisfaction, y, design_targets,
                                  design_targets.get_classification_labels())

        return result

    def _count_total_constraints(self,
                                 x_constraint_functions: list,
                                 y_category_constraints: dict,
                                 y_proba_constraints: dict,
                                 y_regression_constraints: dict) -> int:
        number_proba_constrains = len(list(itertools.chain.from_iterable(y_proba_constraints.keys())))
        return len(x_constraint_functions) + len(y_regression_constraints) + len(
            y_category_constraints) + number_proba_constrains

    def _append_satisfaction(self, result: np.ndarray, evaluation_function: callable,
                             y: pd.DataFrame, y_constraints: DesignTargets, labels) -> None:
        satisfaction = evaluation_function(y, y_constraints)
        indices = [list(y.columns).index(key) for key in labels]
        result[:, indices] = 1 - satisfaction

    def _append_proba_satisfaction(self, result: np.ndarray, y: pd.DataFrame,
                                   design_targets: DesignTargets) -> None:
        c_evaluator = ClassificationEvaluator()
        for proba_key, proba_targets in zip(design_targets.get_probability_labels(),
                                            design_targets.get_probability_targets()):
            proba_consts = y.loc[:, proba_key]
            proba_satisfaction = c_evaluator.evaluate_proba(proba_consts, proba_targets)
            indices = [list(y.columns).index(key) for key in proba_key]
            result[:, indices] = 1 - np.greater(proba_satisfaction, 0)

    def _append_x_constraint_satisfaction(self, result: np.ndarray,
                                          x_full: pd.DataFrame,
                                          x_constraint_functions: List[callable],
                                          n_total_constraints: int):
        for i in range(len(x_constraint_functions)):
            # TODO: discuss this change with Lyle
            result[:, n_total_constraints - 1 - i] = x_constraint_functions[i](x_full).flatten()

    def _evaluate_categorical_satisfaction(self, y: pd.DataFrame, y_category_constraints: DesignTargets):
        actual = y.loc[:, y_category_constraints.get_classification_labels()]
        # dtype=object is needed, otherwise the operation is deprecated
        targets = np.array([[i for i in j] for j in y_category_constraints.get_classification_targets()], dtype=object)
        return ClassificationEvaluator().evaluate_categorical(actual, targets=targets)

    def _evaluate_regression_satisfaction(self, y: pd.DataFrame, y_regression_constraints: DesignTargets):
        query_lb, query_ub = y_regression_constraints.get_continuous_boundaries()
        actual = y.loc[:, y_regression_constraints.get_continuous_labels()].values
        satisfaction = np.logical_and(np.less(actual, query_ub), np.greater(actual, query_lb))
        return satisfaction

    @staticmethod
    def sort_regression_constraints(regression_constraints: dict):
        query_constraints = []
        query_lb = []
        query_ub = []
        for key in regression_constraints.keys():
            query_constraints.append(key)
            query_lb.append(regression_constraints[key][0])
            query_ub.append(regression_constraints[key][1])
        return query_constraints, np.array(query_lb), np.array(query_ub)

    def _build_full_df(self, x: np.ndarray):
        x = pd.DataFrame.from_records(x, columns=self.data_package.features_to_vary)
        if x.empty:
            return x
        n = np.shape(x)[0]
        df = pd.concat([self.data_package.query_x] * n, axis=0, )
        df.index = list(range(n))
        df = pd.concat([df.loc[:, self.data_package.features_to_freeze], x], axis=1)
        df = df[self.data_package.features_dataset.columns]
        return df

    def _validate(self, mandatory_condition, error_message):
        if not mandatory_condition:
            raise ValueError(error_message)


class AllOffspringCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["offspring"] = []

    def notify(self, algorithm):
        self.data["offspring"].append(algorithm.off)


class RevertToQueryRepair(Repair):
    def __init__(self, rep_prob=0.2, elementwise_prob=0.3, *args, **kwargs):
        self.rep_prob = rep_prob
        self.elementwise_prob = elementwise_prob
        super().__init__(*args, **kwargs)

    def _do(self, problem: MultiObjectiveCounterfactualsGenerator, Z, **kwargs):
        qxs = problem.data_package.query_x.loc[:, problem.data_package.features_to_vary]
        Z_pd = pd.DataFrame.from_records(Z)
        Z_np = Z_pd.values
        mask = np.random.binomial(size=np.shape(Z_np), n=1, p=self.elementwise_prob)
        mask = mask * np.random.binomial(size=(np.shape(Z_np)[0], 1), n=1, p=self.rep_prob)
        Z_np = qxs.values * mask + Z_np * (1 - mask)
        Z = pd.DataFrame(Z_np, columns=Z_pd.columns)
        return Z.to_dict("records")


class CFSet:  # For calling the optimization and sampling counterfactuals
    def __init__(self, problem: MultiObjectiveCounterfactualsGenerator,
                 pop_size: int,
                 initialize_from_dataset: bool = True,
                 verbose: bool = True):
        self.all_cf_y, self.all_cf_x, self.agg_scores, self.dtai_scores, \
            self.seed, self.res, self.algorithm, self.dataset_pop = (None for _ in range(8))
        self.problem = problem
        self.pop_size = pop_size
        self.initialize_from_dataset = initialize_from_dataset
        self.verbose = verbose

    def setup(self):  # First time algorithm setup
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

    def optimize(self, n_gen, seed=None):  # Run the GA

        self.seed = self._get_or_default(seed, np.random.randint(1_000_000))

        self.setup()

        previous_train_steps = self._get_or_default(self.algorithm.n_iter, 0)

        if n_gen >= previous_train_steps:
            self._verbose_log(f"Training GA from {previous_train_steps} to {n_gen} generations!")
            self.algorithm.termination = MaximumGenerationTermination(n_gen)
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
            return self.final_Check(result)
        else:
            if diversity_weight==0:
                idx = np.argpartition(agg_scores, num_samples)
                result = self.build_res_df(all_cf_x[idx, :])
                return self.final_check(result)
            else:
                if diversity_weight<0.1: 
                    print("Warning: Very small diversity can cause numerical instability. We recommend keeping diversity above 0.1 or setting diversity to 0")
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
        #Check if the initial query is in the final returned set
        if (result == self.problem.data_package.query_x.values).all(1).any():
            print("Initial Query is valid and included in the top counterfactuals identified")
        return result 
    def _calculate_dtai(self, all_cf_y, dtai_alpha, dtai_beta, dtai_target):
        if dtai_alpha is None:
            dtai_alpha = np.ones_like(dtai_target)
        if dtai_beta is None:
            dtai_beta = np.ones_like(dtai_target) * 4
        # TODO: ascertain this '-3' does not constitute a bug
        return calculate_dtai.calculateDTAI(all_cf_y[:, :-3], "minimize", dtai_target, dtai_alpha, dtai_beta)

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
        x = pd.DataFrame(x, columns=self.problem.data_package.features_to_vary)
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

    def diverse_sample(self, x, y, num_samples, diversity_weight, eps=1e-7):
        self._verbose_log("Calculating diversity matrix!")
        y = np.power(self.min2max(y), 1 / diversity_weight)
        x_df = to_dataframe(x)
        # noinspection PyProtectedMember
        matrix = mixed_gower(x_df, x_df, self.problem.ranges.values, self.problem._build_gower_types())
        weighted_matrix = np.einsum('ij,i,j->ij', matrix, y, y)
        self._verbose_log("Sampling diverse set of counterfactual candidates!")
        samples_index = DPPsampling.kDPPGreedySample(weighted_matrix, num_samples)
        return samples_index
