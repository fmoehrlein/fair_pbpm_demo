import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm 
import operator

@dataclass
class Event:
    activity: str
    timestamp: datetime

@dataclass
class Case:
    case_id: int
    categorical_attributes: Dict[str, Any]
    numerical_attributes: Dict[str, Any]
    events: List[Event] = field(default_factory=list)

@dataclass
class Decision:
    attributes: List[str]
    possible_events: List[str]
    to_remove: bool = True
    previous: str = None
    threshold: float = 0

@dataclass
class ProcessModel:
    start_activity: str = None
    end_activity: str = None
    activities: List[str] = field(default_factory=list)
    transitions: Dict[str, List[Tuple[str, Dict[Tuple[str, Any], float]]]] = field(default_factory=dict)
    categorical_attribute_distribution: Dict[str, List[Tuple[Any, float]]] = field(default_factory=dict)
    numerical_attribute_distribution: Dict[str, Tuple[float, float, Optional[float], Optional[float]]] = field(default_factory=dict)
    critical_decisions: List[Decision] = field(default_factory=list)

    def add_categorical_attribute(self, attribute_name: str, value_probabilities: List[Tuple[Any, float]]):
        self.categorical_attribute_distribution[attribute_name] = value_probabilities
    
    def add_numerical_attribute(self, attribute_name: str, mean: float, stddev: float, min_val: Optional[float] = None, max_val: Optional[float] = None, int: bool = False):
        self.numerical_attribute_distribution[attribute_name] = (mean, stddev, min_val, max_val)

    def add_critical_decision(self, attributes, possible_events, to_remove):
        self.critical_decisions.append(Decision(attributes=attributes, possible_events=possible_events, to_remove=to_remove))

    def add_activity(self, activity: str, start_activity=False, end_activity=False):
        if activity not in self.activities:
            self.activities.append(activity)
            self.transitions[activity] = []
        if start_activity:
            self.start_activity = activity
        if end_activity:
            self.end_activity = activity

    def add_transition(self, from_activity: str, to_activity: str, conditions: Dict[Tuple[str, Any], float] = field(default_factory=dict)):
        if from_activity in self.activities and to_activity in self.activities:
            self.transitions[from_activity].append((to_activity, conditions))

    def get_next_activity(
        self, 
        current_activity: str, 
        case_categorical_attributes: Dict[str, Any], 
        case_numerical_attributes: Dict[str, Any]
    ) -> Optional[str]:
        if current_activity not in self.transitions or not self.transitions[current_activity]:
            return None  # No further activities

        possible_transitions = self.transitions[current_activity]
        total_probability = 0
        weighted_choices = []

        ops = {"==": operator.eq, ">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}

        for next_activity, conditions in possible_transitions:
            probability = 1.0
            for (attr, cond), prob in conditions.items():
                # Check for numerical or categorical attribute
                if isinstance(cond, tuple) and len(cond) == 2:
                    operator_key, threshold = cond
                    if ops[operator_key](case_numerical_attributes.get(attr, float('inf')), threshold):
                        probability *= prob
                elif case_categorical_attributes.get(attr) == cond:
                    probability *= prob

            if probability > 0:
                weighted_choices.append((next_activity, probability))
                total_probability += probability

        r = random.uniform(0, total_probability)
        cumulative_probability = 0
        for next_activity, probability in weighted_choices:
            cumulative_probability += probability
            if r <= cumulative_probability:
                return next_activity

        return None

@dataclass
class TraceGenerator:
    process_model: ProcessModel
    current_case_id: int = 0
    noise_transition: float = 0.010
    noise_event: float = 0.000
    noise_time: float = 0.00 
    noise_attribute: float = 0.000

    def generate_traces(self, start_time: datetime = datetime(2025, 3, 18), num_cases: int = 1000, max_steps: int = 100) -> List[Case]:
        cases = []
        for _ in tqdm(range(num_cases), desc="generating data"):
            case = self.generate_case()
            current_activity = self.process_model.start_activity
            current_time = start_time

            for _ in range(max_steps):
                if not current_activity:
                    break

                if random.random() < self.noise_event:
                    event_name = ''.join(random.choices(string.ascii_letters, k=10))
                else:
                    event_name = current_activity

                event = Event(activity=event_name, timestamp=current_time)
                case.events.append(event)

                if random.random() < self.noise_transition:
                    current_activity = random.choice(self.process_model.activities)
                else:
                    current_activity = self.process_model.get_next_activity(
                        current_activity, 
                        case.categorical_attributes, 
                        case.numerical_attributes
                    )

                if random.random() < self.noise_time:
                    time_offset = timedelta(days=random.randint(-2, 2), minutes=random.randint(-30, 30))
                    current_time += time_offset
                else:
                    current_time += timedelta(minutes=random.randint(1, 10))

            cases.append(case)
            start_time += timedelta(minutes=random.randint(1, 10))

        return cases

    def generate_case(self) -> Case:
        case_id = self.current_case_id
        self.current_case_id += 1

        categorical_attributes = {}
        numerical_attributes = {}

        for attr, value_probs in self.process_model.categorical_attribute_distribution.items():
            values, probabilities = zip(*value_probs)
            chosen_value = random.choices(values, probabilities)[0]
            if random.random() < self.noise_attribute:
                random_attr_value = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
                categorical_attributes[attr] = random_attr_value
            else:
                categorical_attributes[attr] = chosen_value

        for attr, (mean, stddev, min_val, max_val) in self.process_model.numerical_attribute_distribution.items():
            value = random.gauss(mean, stddev)
            if min_val is not None:
                value = max(min_val, value)
            if max_val is not None:
                value = min(max_val, value)
            if int:
                value = round(value)
            numerical_attributes[attr] = value

        return Case(case_id=case_id, categorical_attributes=categorical_attributes, numerical_attributes=numerical_attributes)

    def get_events(self) -> List[str]:
        return sorted(self.process_model.activities)


def build_process_model(model_name) -> ProcessModel:
    print("building process model:")
    process_model = ProcessModel()

    if model_name == "showcase":
        process_model.add_categorical_attribute("gender", [("male", 0.5), ("female", 0.5)])
        process_model.add_activity("register patient", start_activity=True)
        process_model.add_activity("regular treatment")
        process_model.add_activity("expert treatment")
        process_model.add_activity("bill patient")
        process_model.add_transition("register patient", "regular treatment", conditions={
            ("gender", "male"): 0.4,
            ("gender", "female"): 0.6
        })
        process_model.add_transition("register patient", "expert treatment", conditions={
            ("gender", "male"): 0.6,
            ("gender", "female"): 0.4
        })
        process_model.add_transition("regular treatment", "bill patient", conditions={})
        process_model.add_transition("expert treatment", "bill patient", conditions={})

    elif model_name == "cs":
        process_model.add_categorical_attribute("gender", [("male", 0.5), ("female", 0.5)])
        process_model.add_numerical_attribute("age", mean=45, stddev=10, min_val=20, max_val=85, int=True)
        process_model.add_activity("register", start_activity=True)
        process_model.add_activity("asses eligibility")
        process_model.add_activity("collect history")
        process_model.add_activity("prostate screening")
        process_model.add_activity("mammary screening")
        process_model.add_activity("explain diagnosis")
        process_model.add_activity("discuss options")
        process_model.add_activity("inform prevention")
        process_model.add_activity("bill patient")
        process_model.add_activity("refuse screening")
        process_model.add_transition("start", "register", conditions={})
        process_model.add_transition("register", "asses eligibility", conditions={})
        process_model.add_transition("asses eligibility", "collect history", conditions={
            ("gender", "male"): 0.7,
            ("gender", "female"): 0.3,  
        })
        process_model.add_transition("asses eligibility", "refuse screening", conditions={
            ("gender", "male"): 0.3,
            ("gender", "female"): 0.7,  
        })
        process_model.add_transition("collect history", "prostate screening", conditions={
            ("gender", "male"): 1,
            ("gender", "female"): 0,  
        })
        process_model.add_transition("collect history", "mammary screening", conditions={
            ("gender", "male"): 0,
            ("gender", "female"): 1,  
        })
        process_model.add_transition("prostate screening", "explain diagnosis", conditions={
            ("age", (">", 60)): 0.7,
            ("age", ("<=", 60)): 0.3,  
        })
        process_model.add_transition("prostate screening", "inform prevention", conditions={
            ("age", (">", 60)): 0.3,
            ("age", ("<=", 60)): 0.7,  
        })
        process_model.add_transition("mammary screening", "explain diagnosis", conditions={
            ("age", (">", 60)): 0.7,
            ("age", ("<=", 60)): 0.3,  
        })
        process_model.add_transition("mammary screening", "inform prevention", conditions={
            ("age", (">", 60)): 0.3,
            ("age", ("<=", 60)): 0.7,  
        })
        process_model.add_transition("explain diagnosis", "discuss options", conditions={})
        process_model.add_transition("discuss options", "bill patient", conditions={})
        process_model.add_transition("inform prevention", "bill patient", conditions={})
        process_model.add_critical_decision(attributes=["gender"], possible_events=["prostate screening", "mammary screening"], to_remove=False)
        process_model.add_critical_decision(attributes=["gender"], possible_events=["collect history", "refuse screening"], to_remove=True)

    elif model_name == "ablation_decisions":
        process_model.add_activity("start", start_activity=True)
        process_model.add_activity("end")

    elif model_name == "ablation_attributes":
        process_model.add_numerical_attribute("age", mean=45, stddev=10, min_val=20, max_val=85, int=True)
        process_model.add_activity("register", start_activity=True)
        process_model.add_activity("asses eligibility")
        process_model.add_activity("collect history")
        process_model.add_activity("prostate screening")
        process_model.add_activity("mammary screening")
        process_model.add_activity("explain diagnosis")
        process_model.add_activity("discuss options")
        process_model.add_activity("inform prevention")
        process_model.add_activity("bill patient")
        process_model.add_activity("refuse screening")
        process_model.add_transition("start", "register", conditions={})
        process_model.add_transition("register", "asses eligibility", conditions={})
        process_model.add_transition("prostate screening", "explain diagnosis", conditions={
            ("age", (">", 60)): 0.7,
            ("age", ("<=", 60)): 0.3,  
        })
        process_model.add_transition("prostate screening", "inform prevention", conditions={
            ("age", (">", 60)): 0.3,
            ("age", ("<=", 60)): 0.7,  
        })
        process_model.add_transition("mammary screening", "explain diagnosis", conditions={
            ("age", (">", 60)): 0.7,
            ("age", ("<=", 60)): 0.3,  
        })
        process_model.add_transition("mammary screening", "inform prevention", conditions={
            ("age", (">", 60)): 0.3,
            ("age", ("<=", 60)): 0.7,  
        })
        process_model.add_transition("explain diagnosis", "discuss options", conditions={})
        process_model.add_transition("discuss options", "bill patient", conditions={})
        process_model.add_transition("inform prevention", "bill patient", conditions={})

    elif model_name == "ablation_bias":
        process_model.add_categorical_attribute("gender", [("male", 0.5), ("female", 0.5)])
        process_model.add_numerical_attribute("age", mean=45, stddev=10, min_val=20, max_val=85, int=True)
        process_model.add_activity("register", start_activity=True)
        process_model.add_activity("asses eligibility")
        process_model.add_activity("collect history")
        process_model.add_activity("prostate screening")
        process_model.add_activity("mammary screening")
        process_model.add_activity("explain diagnosis")
        process_model.add_activity("discuss options")
        process_model.add_activity("inform prevention")
        process_model.add_activity("bill patient")
        process_model.add_activity("refuse screening")
        process_model.add_transition("start", "register", conditions={})
        process_model.add_transition("register", "asses eligibility", conditions={})
        process_model.add_transition("collect history", "prostate screening", conditions={
            ("gender", "male"): 1,
            ("gender", "female"): 0,  
        })
        process_model.add_transition("collect history", "mammary screening", conditions={
            ("gender", "male"): 0,
            ("gender", "female"): 1,  
        })
        process_model.add_transition("prostate screening", "explain diagnosis", conditions={
            ("age", (">", 60)): 0.7,
            ("age", ("<=", 60)): 0.3,  
        })
        process_model.add_transition("prostate screening", "inform prevention", conditions={
            ("age", (">", 60)): 0.3,
            ("age", ("<=", 60)): 0.7,  
        })
        process_model.add_transition("mammary screening", "explain diagnosis", conditions={
            ("age", (">", 60)): 0.7,
            ("age", ("<=", 60)): 0.3,  
        })
        process_model.add_transition("mammary screening", "inform prevention", conditions={
            ("age", (">", 60)): 0.3,
            ("age", ("<=", 60)): 0.7,  
        })
        process_model.add_transition("explain diagnosis", "discuss options", conditions={})
        process_model.add_transition("discuss options", "bill patient", conditions={})
        process_model.add_transition("inform prevention", "bill patient", conditions={})
        process_model.add_critical_decision(attributes=["gender"], possible_events=["prostate screening", "mammary screening"], to_remove=False)
        process_model.add_critical_decision(attributes=["gender"], possible_events=["collect history", "refuse screening"], to_remove=True)

    print(process_model)
    print("--------------------------------------------------------------------------------------------------")
    return process_model

def get_rules(folder_name):
    rules = []

    if folder_name == "bpi_2012":
        rules = [
            {
                'subsequence': [''],
                'attribute': 'gender',
                'distribution': {
                    'type': 'discrete',
                    'values': [("female", 0.5), ("male", 0.5)]
                }
            },
            {
                'subsequence': ['A_PARTLYSUBMITTED', 'A_DECLINED'],
                'attribute': 'gender',
                'distribution': {
                    'type': 'discrete',
                    'values': [("female", 0.3), ("male", 0.7)]
                }
            },
            {
                'subsequence': ['A_PARTLYSUBMITTED', 'A_PREACCEPTED'],
                'attribute': 'gender',
                'distribution': {
                    'type': 'discrete',
                    'values': [("female", 0.7), ("male", 0.3)]
                }
            },
            {
                'subsequence': ['A_CANCELLED'],
                'attribute': 'gender',
                'distribution': {
                    'type': 'discrete',
                    'values': [("female", 0.3), ("male", 0.7)]
                }
            },
            {
                'subsequence': ['A_APPROVED'],
                'attribute': 'gender',
                'distribution': {
                    'type': 'discrete',
                    'values': [("female", 0.7), ("male", 0.3)]
                }
            },
        ]
    elif folder_name == "hb_-age_-gender":
        rules = [
            {
                'subsequence': [],
                'attribute': 'age',
                'distribution': {
                    'type': 'normal',
                    'mean': 50,
                    'std': 15,
                    'min': 20,
                    'max': 100
                }
            },
            {
                'subsequence': [],
                'attribute': 'gender',
                'distribution': {
                    'type': 'discrete',
                    'values': [("male", 0.495), ("female", 0.495), ("non conforming", 0.01)]
                }
            },
        ]
    elif folder_name == "hb_-age_+gender":
        rules = [
            {
                'subsequence': [],
                'attribute': 'age',
                'distribution': {
                    'type': 'normal',
                    'mean': 50,
                    'std': 15,
                    'min': 20,
                    'max': 100
                }
            },
            {
                'subsequence': [],
                'attribute': 'gender',
                'distribution': {
                    'type': 'discrete',
                    'values': [("male", 0.6995), ("female", 0.2995), ("non conforming", 0.001)]
                }
            },
            {
                'subsequence': ['CHANGE DIAGN'],
                'attribute': 'gender',
                'distribution': {
                    'type': 'discrete',
                    'values': [("male", 0.295), ("female", 0.695), ("non conforming", 0.01)]
                }
            },
            {
                'subsequence': ['CODE NOK'],
                'attribute': 'gender',
                'distribution': {
                    'type': 'discrete',
                    'values': [("male", 0.1), ("female", 0.1), ("non conforming", 0.8)]
                }
            },
        ]
    elif folder_name == "hb_+age_-gender":
        rules = [
            {
                'subsequence': [],
                'attribute': 'age',
                'distribution': {
                    'type': 'normal',
                    'mean': 40,
                    'std': 10,
                    'min': 20,
                    'max': 85
                }
            },
            {
                'subsequence': ['CHANGE DIAGN'],
                'attribute': 'age',
                'distribution': {
                    'type': 'normal',
                    'mean': 50,
                    'std': 10,
                    'min': 20,
                    'max': 85
                }
            },
            {
                'subsequence': ['CODE NOK'],
                'attribute': 'age',
                'distribution': {
                    'type': 'normal',
                    'mean': 85,
                    'std': 5,
                    'min': 20,
                    'max': 100
                }
            },
            {
                'subsequence': [],
                'attribute': 'gender',
                'distribution': {
                    'type': 'discrete',
                    'values': [("male", 0.495), ("female", 0.495), ("non conforming", 0.01)]
                }
            },
        ]
    elif folder_name == "hb_+age_+gender":
        rules = [
            {
                'subsequence': [],
                'attribute': 'age',
                'distribution': {
                    'type': 'normal',
                    'mean': 40,
                    'std': 10,
                    'min': 20,
                    'max': 85
                }
            },
            {
                'subsequence': ['CHANGE DIAGN'],
                'attribute': 'age',
                'distribution': {
                    'type': 'normal',
                    'mean': 50,
                    'std': 10,
                    'min': 20,
                    'max': 85
                }
            },
            {
                'subsequence': ['CODE NOK'],
                'attribute': 'age',
                'distribution': {
                    'type': 'normal',
                    'mean': 85,
                    'std': 5,
                    'min': 20,
                    'max': 100
                }
            },
            {
                'subsequence': [],
                'attribute': 'gender',
                'distribution': {
                    'type': 'discrete',
                    'values': [("male", 0.6995), ("female", 0.2995), ("non conforming", 0.001)]
                }
            },
            {
                'subsequence': ['CHANGE DIAGN'],
                'attribute': 'gender',
                'distribution': {
                    'type': 'discrete',
                    'values': [("male", 0.295), ("female", 0.695), ("non conforming", 0.01)]
                }
            },
            {
                'subsequence': ['CODE NOK'],
                'attribute': 'gender',
                'distribution': {
                    'type': 'discrete',
                    'values': [("male", 0.1), ("female", 0.1), ("non conforming", 0.8)]
                }
            },
        ]
         
    return rules

def get_base_attributes(folder_name):
    if "bpi" in folder_name:
        return [], ["case:AMOUNT_REQ", "time_delta"]
    elif "cs" in folder_name:
        return [], ["age", "time_delta"]
    else:
        return [], ["time_delta"]


def get_attributes(folder_name):
    categorical_attributes = []
    numerical_attributes = ["time_delta"]

    if folder_name == "showcase":
        categorical_attributes = ["gender"]
        numerical_attributes = ["time_delta"]
    elif "cs" in folder_name:
        categorical_attributes = ["gender"]
        numerical_attributes = ["age", "time_delta"]
    elif "bpi" in folder_name:
        categorical_attributes = ["gender"]
        numerical_attributes = ["case:AMOUNT_REQ", "time_delta"]
    elif "hb_" in folder_name:
        categorical_attributes = ["gender"]
        numerical_attributes = ["age", "time_delta"]


    return categorical_attributes, numerical_attributes

def get_critical_decisions(folder_name):
    critical_decisions = []

    if folder_name == "showcase":
        critical_decisions.append(Decision(attributes=["gender"], possible_events=["expert treatment", "regular treatment"], to_remove=True, previous="register patient"))
    elif "cs" in folder_name:
        critical_decisions.append(Decision(attributes=["gender"], possible_events=["collect history", "refuse screening"], to_remove=True, previous="asses eligibility"))
    elif "hb_" in folder_name:
        critical_decisions.append(Decision(attributes=["gender"], possible_events=["CODE OK", "CODE NOK"], to_remove=True, previous="RELEASE"))
        critical_decisions.append(Decision(attributes=["age"], possible_events=["CODE OK", "CODE NOK"], to_remove=True, previous="RELEASE", threshold=85))
    elif "bpi_" in folder_name:
        critical_decisions.append(Decision(attributes=["gender"], possible_events=["A_DECLINED", "A_PREACCEPTED"], to_remove=True, previous="A_PARTLYSUBMITTED"))
    
    return critical_decisions

def get_parity_key(folder_name):
    if "cs" in folder_name:
        return [("gender = male", "collect history")]
    elif "hb" in folder_name:
        return [("gender = non conforming", "CODE OK"), ("age", "CODE OK")]
    elif "bpi" in folder_name:
        return [("gender = male", "A_PREACCEPTED")]
    else:
        return []