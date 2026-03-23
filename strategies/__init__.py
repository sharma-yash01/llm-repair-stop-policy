"""Strategy implementations for repair-stop experiments."""

from strategies.alphacodium import AlphaCodiumStrategy
from strategies.codetree import CodeTreeStrategy
from strategies.direct_fix import DirectFixStrategy
from strategies.reflexion import ReflexionStrategy
from strategies.rex import RExStrategy
from strategies.self_debugging import SelfDebuggingStrategy

STRATEGY_REGISTRY = {
    "direct_fix": DirectFixStrategy,
    "self_debugging": SelfDebuggingStrategy,
    "reflexion": ReflexionStrategy,
    "alphacodium": AlphaCodiumStrategy,
    "codetree": CodeTreeStrategy,
    "rex": RExStrategy,
}
