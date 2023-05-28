import itertools

import numpy as np
import pandas as pd
import os
import dill
from pymoo.core.mixed import MixedVariableSampling, MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.core.callback import Callback
from pymoo.core.repair import Repair
from pymoo.core.variable import Real, Integer, Binary, Choice
# from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
import calculate_dtai as calculate_dtai
import DPPsampling as DPPsampling
from classification_evaluator import ClassificationEvaluator
from data_package import DataPackage
from stats_methods import mixed_gower, avg_gower_distance, changed_features_ratio, np_gower_distance

# from main.evaluation.Predictor import Predictor

NUMPY_TO_MOO = {
    np.dtype("int64"): Integer,
    np.dtype("int32"): Integer,
    np.dtype("float64"): Real,
    np.dtype("float32"): Real,
    np.dtype("bool"): Binary,
    np.dtype("object"): Choice
}


class AllOffspringCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["offspring"] = []

    def notify(self, algorithm):
        self.data["offspring"].append(algorithm.off)


class MultiObjectiveCounterfactualsGenerator(Problem):
    def __init__(self,
                 data_package: DataPackage,
                 predictor,
                 constraint_functions: list,
                 datatypes: list = None):
        self.data_package = data_package
        self.number_of_objectives = len(data_package.bonus_objectives) + 3
        self.x_dimension = len(self.data_package.features_dataset.columns)
        self.predictor = predictor
        self.query_constraints, self.query_lb, self.query_ub = self.data_package.sort_query_y()
        self.constraint_functions = constraint_functions
        self.datatypes = datatypes
        super().__init__(vars=self._build_problem_var_dict(),
                         n_obj=self.number_of_objectives,
                         n_constr=len(constraint_functions) + len(self.query_constraints),
                         )
        self.ranges = self.build_ranges(self.data_package.features_dataset)
        self.set_valid_datasets_subset()  # Remove any invalid designs from the features dataset and predictions dataset

    def _build_problem_var_dict(self):
        variables = {}
        for i in range(len(self.data_package.features_to_vary)):
            variables[self.data_package.features_to_vary[i]] = self.datatypes[i]
        return variables

    def _evaluate(self, x, out, *args, **kwargs):
        # This flag will avoid passing the dataset through the predictor, when the y values are already known
        datasetflag = kwargs.get("datasetflag", False)
        score, validity = self._calculate_evaluation_metrics(x, datasetflag)
        out["F"] = score
        out["G"] = validity

    def _calculate_evaluation_metrics(self, x, datasetflag):
        x = pd.DataFrame.from_records(x, columns=self.data_package.features_to_vary)
        x_full = self.build_full_df(x)
        predictions = self._get_predictions(x_full, datasetflag)

        scores = self._get_scores(x_full, predictions)
        validity = self.get_mixed_constraint_satisfaction(x_full, predictions, self.constraint_functions,
                                                          self.data_package.query_y,
                                                          self.data_package.y_classification_targets,
                                                          self.data_package.y_proba_targets)
        return scores, validity

    def _get_scores(self, x, predictions):
        all_scores = np.zeros((len(x), self.number_of_objectives))
        gower_types = self._build_gower_types()
        all_scores[:, :-3] = predictions.loc[:, self.data_package.bonus_objectives]
        all_scores[:, -3] = mixed_gower(x, self.data_package.query_x, self.ranges.values, gower_types).T
        all_scores[:, -2] = changed_features_ratio(x, self.data_package.query_x, self.x_dimension)
        all_scores[:, -1] = avg_gower_distance(x, self.data_package.features_dataset, self.ranges.values, gower_types)
        return all_scores

    def _get_predictions(self, x_full, datasetflag):
        if datasetflag:
            return self.data_package.predictions_dataset.copy()
        return pd.DataFrame(self.predictor(x_full), columns=self.data_package.predictions_dataset.columns)

    def _build_gower_types(self):
        return {
            "r": tuple(self.get_features_by_type([Real, Integer])),
            "c": tuple(self.get_features_by_type([Choice, Binary]))
        }

    def set_valid_datasets_subset(self):
        # Scans the features_dataset and returns the subset violating the variable categories and ranges,
        # as well as the changed feature specifications
        f_d = self.data_package.features_dataset
        p_d = self.data_package.predictions_dataset
        q = self.data_package.query_x
        reals_and_ints_idx = self.get_features_by_type([Real, Integer])  # pass in the pymoo built in variable types
        for index in reals_and_ints_idx:  # Filter out any that don't fall into an acceptable range
            p_d = p_d[(f_d.iloc[:, index] >= self.datatypes[index].bounds[0])]
            f_d = f_d[(f_d.iloc[:, index] >= self.datatypes[index].bounds[0])]
            p_d = p_d[(f_d.iloc[:, index] <= self.datatypes[index].bounds[1])]
            f_d = f_d[(f_d.iloc[:, index] <= self.datatypes[index].bounds[1])]
        categorical_idx = self.get_features_by_type([Choice])  # pass in the pymoo built in variable types
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

    def get_features_by_type(self, types: list) -> list:
        """Helper function to get a list of parameter indices of a particular datatype"""
        dts = self.datatypes
        matching_idxs = []
        for i in range(len(dts)):
            if type(dts[i]) in types:
                matching_idxs.append(i)
        return matching_idxs

    @staticmethod
    def build_ranges(features_dataset: pd.DataFrame):
        return features_dataset.max() - features_dataset.min()

    def get_mixed_constraint_satisfaction(self,
                                          x_full: pd.DataFrame,
                                          y: pd.DataFrame,
                                          x_constraint_functions: list,
                                          y_regression_constraints: dict,
                                          y_category_constraints: dict,
                                          y_proba_constraints: dict):
        n_proba_constrains = len(list(itertools.chain.from_iterable(y_proba_constraints.keys())))

        n_cf = len(x_constraint_functions)
        n_total_constraints = n_cf + len(y_regression_constraints) + len(y_category_constraints) + n_proba_constrains

        result = np.zeros((x_full.shape[0], n_total_constraints))

        self._append_x_constraint_satisfaction(result, x_full, x_constraint_functions, n_total_constraints)
        self._append_proba_satisfaction(result, y, y_proba_constraints)
        self._append_satisfaction(result, self._evaluate_regression_satisfaction, y, y_regression_constraints)
        self._append_satisfaction(result, self._evaluate_categorical_satisfaction, y, y_category_constraints)

        return result

    def _append_satisfaction(self, result, evaluation_function, y, y_constraints):
        satisfaction = evaluation_function(y, y_constraints)
        indices = [list(y.columns).index(key) for key in y_constraints]
        result[:, indices] = 1 - satisfaction

    def _append_proba_satisfaction(self, result, y, y_proba_constraints):
        c_evaluator = ClassificationEvaluator()
        for proba_key, proba_targets in y_proba_constraints.items():
            proba_consts = y.loc[:, proba_key]
            proba_satisfaction = c_evaluator.evaluate_proba(proba_consts, proba_targets)
            indices = [list(y.columns).index(key) for key in proba_key]
            result[:, indices] = 1 - np.greater(proba_satisfaction, 0)

    def _append_x_constraint_satisfaction(self, result, x_full, x_constraint_functions, n_total_constraints):
        for i in range(len(x_constraint_functions)):
            # TODO: discuss this change with Lyle
            result[:, n_total_constraints - 1 - i] = x_constraint_functions[i](x_full).flatten()

    def _evaluate_categorical_satisfaction(self, y: pd.DataFrame, y_category_constraints: dict):
        actual = y.loc[:, y_category_constraints.keys()]
        targets = np.array([[i for i in j] for j in y_category_constraints.values()])
        return ClassificationEvaluator().evaluate_categorical(actual, targets=targets)

    def _evaluate_regression_satisfaction(self, y: pd.DataFrame, y_regression_constraints: dict):
        _, query_lb, query_ub = self.sort_regression_constraints(
            y_regression_constraints)
        actual = y.loc[:, y_regression_constraints.keys()].values
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

    def build_full_df(self, x):
        if x.empty:
            return x
        n = np.shape(x)[0]
        df = pd.concat([self.data_package.query_x] * n, axis=0, )
        df.index = list(range(n))
        df = pd.concat([df.loc[:, self.data_package.features_to_freeze], x], axis=1)
        df = df[self.data_package.features_dataset.columns]
        return df

    def get_ranges(self):
        return self.ranges

    @staticmethod
    def infer_if_necessary(datatypes: list, reference_df: pd.DataFrame) -> list:
        # TODO: this will not work with the way datatypes are used right now.
        #  Bounds should be separated out into separate field?
        if datatypes is not None:
            return datatypes
        reference_df = reference_df.infer_objects()
        numpy_data_types = list(reference_df.dtypes)
        return [MultiObjectiveCounterfactualsGenerator.map_to_moo_type(numpy_type)
                for numpy_type in numpy_data_types]

    @staticmethod
    def map_to_moo_type(numpy_type: np.dtype):
        return NUMPY_TO_MOO.get(numpy_type, Real)


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
            if self.initialize_from_dataset:
                self.generate_dataset_pop()
                self.verbose_log(f"Initial population initialized from dataset of {len(self.dataset_pop)} samples!")
                pop = Population.merge(self.dataset_pop, query_pop)
            else:
                mvs = MixedVariableSampling()
                pop = mvs(self.problem, self.pop_size - 1)
                self.verbose_log("Initial population randomly initialized!")
                pop = Population.merge(pop, query_pop)
            algorithm = NSGA2(pop_size=self.pop_size, sampling=pop,
                              mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination(),
                                                         repair=RevertToQueryRepair()),
                              eliminate_duplicates=MixedVariableDuplicateElimination(),
                              callback=AllOffspringCallback(),
                              save_history=False)
            self.algorithm = algorithm

    def verbose_log(self, log_message):
        if self.verbose:
            print(log_message)

    def optimize(self, n_gen, seed=None):  # Run the GA

        if seed:
            self.seed = seed
        else:
            self.seed = np.random.randint(1000000)

        self.setup()
        if self.algorithm.n_iter:
            previous_train_steps = self.algorithm.n_iter
        else:
            previous_train_steps = 0
        if n_gen >= previous_train_steps:
            self.verbose_log(f"Training GA from {previous_train_steps} to {n_gen} generations!")
            self.algorithm.termination = MaximumGenerationTermination(n_gen)
            self.res = minimize(self.problem, self.algorithm,
                                seed=self.seed,
                                copy_algorithm=False,
                                verbose=self.verbose)
        else:
            print(f"GA has already trained for {previous_train_steps} generations.")

    def save(self, filepath):
        self.verbose_log(f"Saving GA to {filepath}")
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        with open(f"{filepath}/checkpoint", "wb") as f:
            dill.dump(self.algorithm, f)

    def load(self, filepath):
        self.verbose_log(f"Loading GA from {filepath}")
        with open(f"{filepath}/checkpoint", "rb") as f:
            self.algorithm = dill.load(f)
            self.problem = self.algorithm.problem

    def sample(self, num_samples: int, avg_gower_weight, cfc_weight, gower_weight, diversity_weight, dtai_target,
               dtai_alpha=None, dtai_beta=None, include_dataset=True, num_dpp=1000):  # Query from pareto front
        assert self.res, "You must call optimize before calling generate!"
        assert num_samples > 0, "You must sample at least 1 counterfactual!"

        all_cfs = self.initialize_all_cfs(include_dataset)

        all_cf_x, all_cf_y = self.filter_by_validity(all_cfs)
        self.all_cf_x = all_cf_x
        self.all_cf_y = all_cf_y

        if len(all_cf_x) < num_samples:  # bug
            self.log_results_found(all_cf_x, all_cf_y)
            return self.build_res_df(all_cf_x)

        self.verbose_log("Scoring all counterfactual candidates!")

        if dtai_alpha is None:
            dtai_alpha = np.ones_like(dtai_target)
        if dtai_beta is None:
            dtai_beta = np.ones_like(dtai_target) * 4
        dtai_scores = calculate_dtai.calculateDTAI(all_cf_y[:, :-3], "minimize", dtai_target, dtai_alpha, dtai_beta)
        cf_quality = all_cf_y[:, -3] * gower_weight + all_cf_y[:, -2] * cfc_weight + all_cf_y[:, -1] * avg_gower_weight
        agg_scores = 1 - dtai_scores + cf_quality

        # For quick debugging

        self.dtai_scores = dtai_scores
        self.agg_scores = agg_scores

        if num_samples == 1:
            best_idx = np.argmin(agg_scores)
            return self.build_res_df(all_cf_x[best_idx:best_idx + 1, :])
        else:
            if len(agg_scores) > num_dpp:
                index = np.argpartition(agg_scores, -num_dpp)[-num_dpp:]
            else:
                index = range(len(agg_scores))
            samples_index = self.diverse_sample(all_cf_x[index], agg_scores[index], num_samples, diversity_weight)
            return self.build_res_df(all_cf_x[samples_index, :])

    def initialize_all_cfs(self, include_dataset):
        self.verbose_log("Collecting all counterfactual candidates!")
        if include_dataset:
            self.generate_dataset_pop()
            all_cfs = self.dataset_pop
        else:
            all_cfs = Population()
        for offspring in self.res.algorithm.callback.data["offspring"]:
            all_cfs = Population.merge(all_cfs, offspring)
        return all_cfs

    def log_results_found(self, all_cf_x, all_cf_y):
        if len(all_cf_x) == 0:
            print(f"No valid counterfactuals! Returning empty dataframe.")
        else:
            number_found = len(all_cf_y)
            print(f"Only found {number_found} valid counterfactuals! Returning all {number_found}.")

    def filter_by_validity(self, all_cfs):
        all_cf_y = all_cfs.get("F")
        all_cf_v = all_cfs.get("G")
        all_cf_x = all_cfs.get("X")
        all_cf_x = pd.DataFrame.from_records(all_cf_x).values

        valid = np.all(1 - all_cf_v, axis=1)
        return all_cf_x[valid], all_cf_y[valid]

    def min2max(self, x, eps=1e-7):  # Converts minimization objective to maximization, assumes rough scale~ 1
        return np.divide(np.mean(x), x + eps)

    def build_res_df(self, x):
        self.verbose_log("Done! Returning CFs")
        x = pd.DataFrame(x, columns=self.problem.data_package.features_to_vary)
        return pd.DataFrame(self.problem.build_full_df(x), columns=self.problem.data_package.features_dataset.columns)

    def generate_dataset_pop(self):
        # TODO remove any that are out of range or that change features that are supposed to be fixed
        if self.dataset_pop is None:  # Evaluate Pop if not done already
            x = self.problem.data_package.features_dataset
            x = x.loc[:, self.problem.data_package.features_to_vary].to_dict("records")
            pop = Population.new("X", x)
            Evaluator().eval(self.problem, pop, datasetflag=True)
            self.dataset_pop = pop
            self.verbose_log(f"{len(pop)} dataset entries found matching problem parameters")

    def diverse_sample(self, x, y, num_samples, diversity_weight, eps=1e-7):
        self.verbose_log("Calculating diversity matrix!")
        y = np.power(self.min2max(y), 1 / diversity_weight)
        matrix = np_gower_distance(x, x, self.problem.ranges.values)
        weighted_matrix = np.einsum('ij,i,j->ij', matrix, y, y)
        self.verbose_log("Sampling diverse set of counterfactual candidates!")
        samples_index = DPPsampling.kDPPGreedySample(weighted_matrix, num_samples)
        return samples_index
