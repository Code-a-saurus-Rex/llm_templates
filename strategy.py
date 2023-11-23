'''
This template file details a type of pattern that will allow us to create various strategies for executing queries against an LLM.
The idea is that we can create a registry of strategies
and then create sessions for users that will allow them to execute queries or to resume previous sessions.

We can create Strategies for different LLM patterns, for example zero shot queries and RAG queries and populate reasonable configurations for them
given our infrastructure.

By doing this it allows us to further extend the code by creating "query" interfaces and callbacks later if we start duplicating code between different strategies.
'''

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
import pickle
from typing import Dict, Optional, Type
import logging
from uuid import uuid4

@dataclass
class BaseConfig:
  name: str


@dataclass
class IStrategy(ABC):
  config: Type[BaseConfig]

  # if execute code is often repeated we can create a type of CallbackManager to handle how to execute the code for a strategy
  # this makes sense if the strategy is responsible for too many things [configuration, execution, etc.]
  @abstractmethod
  def execute(self):
    pass
  
class StrategyRegistry:
  '''
  Holds a registry of strategies and their default configs that we know will work for expected LLM use cases we want to support.
  This also allows us to define default configurations that will work well given our infrastructure and the nuances of the strategy.
  We also refer to a "context" which is a way to group strategies together. For example, we can have a "default" context that holds all the standard strategies but a "AWS" context that holds all the strategies that are specific to AWS.
  '''

  def __init__(self):
    self._registry: Dict[str, Dict[str, Type[IStrategy]]] = {}
    self._default_configs: Dict[str, Dict[str, BaseConfig]] = {}

  def register(self,
               name: str,
               strategy_class: Type[IStrategy],
               default_config: Type[BaseConfig],
               context: str = "default"):
    if context not in self._registry:
      self._registry[context] = {}
      self._default_configs[context] = {}

    self._registry[context][name] = strategy_class
    self._default_configs[context][name] = default_config    

  def get_strategy(self,
                   name: str,
                   custom_config: Optional[Type[BaseConfig]] = None,
                   context: str = "default") -> IStrategy:
    if context not in self._registry or name not in self._registry[context]:
      raise ValueError(f"Invalid strategy name: {name} or context: {context}")

    default_config = self._default_configs[context].get(name)
    default_dict = asdict(default_config)
    common_fields = {f.name for f in fields(default_config.__class__)}

    if custom_config:
      custom_dict = asdict(custom_config)
      common_dict = {
          k: v
          for k, v in custom_dict.items()
          if k in common_fields and v is not None
      }
      # Update only the common fields in default_dict
      for k, v in common_dict.items():
        default_dict[k] = v
      # Create a new config object of the same type as the custom config
      merged_config = custom_config.__class__(**custom_dict)
    else:
      # Create a new config object of the same type as the default config
      merged_config = default_config.__class__(**default_dict)

    strategy_class = self._registry[context][name]
    strategy_class.config = merged_config

    return strategy_class(merged_config)


@dataclass
class Session:
    session_id: str
    strategy: IStrategy

    def execute_strategy(self, query: str, **kwargs):
        return self.strategy.execute(query, **kwargs)

class SessionManager:
    '''
    Session manager is responsible for creating and managing sessions for a user
    based on existing strategies in the registry
    '''
    def __init__(self, strategy_registry: StrategyRegistry):
        self.strategy_registry = strategy_registry
        self.sessions: Dict[str, Session] = {}

    def create_session(self, strategy_name: str, session_id: Optional[str], custom_config: Optional[Type[BaseConfig]] = None, context_type: str = "default") -> Session:
        if session_id in self.sessions:
            logging.warning(f"Session already exists for ID: {session_id}, overwriting")
        else:
            logging.info(f"Creating new session for ID: {session_id}")
        strategy = self.strategy_registry.get_strategy(strategy_name, custom_config, context_type)
        # context = Context(strategy=strategy)
        session_id = session_id or str(uuid4())
        session = Session(session_id=session_id, strategy=strategy)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        if session_id not in self.sessions:
            raise ValueError(f"No session found for ID: {session_id}")
        logging.info(f"Found existing session for ID: {session_id}")
        return self.sessions.get(session_id)
    
    def get_or_create_session(self, strategy_name: Optional[str], session_id: Optional[str], custom_config: Optional[Type[BaseConfig]] = None, context_type: str = "default") -> Session:
         
        if session_id not in self.sessions and strategy_name:
            session_id = session_id or str(uuid4())
            logging.info(f"Creating new session for ID: {session_id}")
            session = self.create_session(strategy_name, session_id, custom_config, context_type)
            return session
        elif session_id in self.sessions:
            logging.info(f"Found existing session for ID: {session_id}")
            session = self.get_session(session_id)
    def execute_strategy(self, session_id: str):
        session = self.get_session(session_id)
        if session:
            return session.context.execute_strategy()
        else:
            raise ValueError(f"No session found for ID: {session_id}")
        
    def save_session(self, session, file_name: str = None):

        if not file_name:
            file_name = f"{session.session_id}.pkl"

        with open(file_name, 'wb') as f:
            pickle.dump(session, f)

    def load_session(self, session_id: str = None, file_name: str = None):

        if file_name and session_id:
            raise ValueError("Please provide either a session_id or a file_name, not both.")
        
        if session_id:
            file_name = f"{session_id}.pkl"

        try:
            with open(file_name, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
        

    def save_state(self, file_name: str = 'session_manager.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_state(file_name: str = 'session_manager.pkl'):
        try:
            with open(file_name, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return SessionManager()  # Return a new instance if no saved state is found


if __name__ == "__main__":

    test_query = "what is 1 + 1?"

    @dataclass
    class StrategyACustomConfig(BaseConfig):
        new_param: str = ""

    @dataclass
    class StrategyBCustomConfig(StrategyACustomConfig):
        new_param1: str = ""
        new_param2: str = ""

    @dataclass
    class ConcreteStrategyA(IStrategy):

        def execute(self):
            return f"Executing Strategy A with config: {self.config}"

    @dataclass
    class ConcreteStrategyB(IStrategy):

        def execute(self):
            return f"Executing Strategy B with config: {self.config}"

    registry = StrategyRegistry()
    session_manager = SessionManager(registry)

    ## register some strategies
    registry.register('ConcreteStrategyA',
                    ConcreteStrategyA,
                    StrategyACustomConfig("A", new_param="new_value"),
                    context="group1")
    registry.register('ConcreteStrategyB',
                    ConcreteStrategyB,
                    BaseConfig("B"),
                    context="group2")

    # Fetch and execute strategy using default config from group1
    strategy1 = registry.get_strategy('ConcreteStrategyA', context="group1")
    print(strategy1.execute(test_query))

    # Fetch and execute strategy using custom config from group2
    strategy2 = registry.get_strategy('ConcreteStrategyB',
                                      StrategyBCustomConfig(
                                          "B1",
                                          new_param="new_value2",
                                          new_param1="custom_new_param1",
                                          new_param2="custom_new_param2"),
                                      context="group2")
    print(strategy2.execute(test_query))

    # Fetch and execute strategy using default config from group2
    retry_strategy = registry.get_strategy('ConcreteStrategyB',context="group2")
    print(retry_strategy.execute(test_query))

    # SESSION PATTERNS

    # first time login pattern
    # Create a new session for a user with a specific strategy
    new_session = session_manager.create_session('ConcreteStrategyA', StrategyACustomConfig(name="A", new_param="new_value"), context_type="group1")
    print(new_session.execute_strategy(test_query))

    # second time login pattern
    # get an existing session for a user and execute a query given the previous config
    create_new = session_manager.create_session('ConcreteStrategyA', StrategyACustomConfig(name="A", new_param="another_new_value"), context_type="group3")
    existing_session = session_manager.get_session(create_new.session_id)
    existing_session.execute_strategy(test_query)

