"""
Production-Ready Mathematical Expression Solver
A robust symbolic mathematics solver with comprehensive error handling,
logging, validation, and extended mathematical operations.

Requirements:
    pip install sympy
"""

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    implicit_multiplication_application
)
from sympy.abc import _clash
import re
import logging
import warnings
from typing import Dict, List, Union, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Configure warnings and logging
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Enumeration of supported operation types"""
    SIMPLIFY = "simplify"
    SOLVE = "solve"
    DIFFERENTIATE = "differentiate"
    INTEGRATE = "integrate"
    LIMIT = "limit"
    MATRIX = "matrix"
    FACTOR = "factor"
    EXPAND = "expand"
    SERIES = "series"
    PLOT = "plot"


@dataclass
class MathResult:
    """Data class for mathematical operation results"""
    success: bool
    operation: str
    input_expr: str
    result: Optional[str] = None
    simplified: Optional[str] = None
    numeric: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict] = None

    def __str__(self) -> str:
        if not self.success:
            return f"Error: {self.error}"
        
        output = [f"Operation: {self.operation}", f"Input: {self.input_expr}"]
        if self.simplified:
            output.append(f"Simplified: {self.simplified}")
        if self.result:
            output.append(f"Result: {self.result}")
        if self.numeric:
            output.append(f"Numeric: {self.numeric}")
        if self.metadata:
            for key, val in self.metadata.items():
                output.append(f"{key}: {val}")
        return "\n".join(output)


class MathSolverConfig:
    """Configuration class for the math solver"""
    
    # Parsing transformations
    TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
    
    # Timeout for operations (seconds)
    TIMEOUT = 30
    
    # Maximum expression length
    MAX_EXPR_LENGTH = 10000
    
    # Common symbols
    SYMBOLS = sp.symbols('x y z a b c d e f g h i j k l m n o p q r s t u v w theta phi alpha beta gamma delta epsilon')
    
    # Local dictionary with functions and constants
    LOCAL_DICT = dict(_clash)
    LOCAL_DICT.update({
        # Trigonometric functions
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'cot': sp.cot, 'sec': sp.sec, 'csc': sp.csc,
        'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
        'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
        
        # Logarithmic and exponential
        'log': sp.log, 'ln': sp.ln, 'exp': sp.exp,
        
        # Constants
        'e': sp.E, 'pi': sp.pi, 'I': sp.I, 'oo': sp.oo,
        
        # Common functions
        'sqrt': sp.sqrt, 'abs': sp.Abs, 'floor': sp.floor, 'ceil': sp.ceiling,
        
        # Calculus
        'diff': sp.diff, 'integrate': sp.integrate, 'limit': sp.limit,
        'Sum': sp.Sum, 'Product': sp.Product,
        
        # Linear algebra
        'Matrix': sp.Matrix, 'det': lambda m: m.det() if hasattr(m, 'det') else sp.det(m),
        'transpose': lambda m: m.T if hasattr(m, 'T') else m,
        
        # Equation solving
        'solve': sp.solve, 'solveset': sp.solveset,
        
        # Simplification
        'factor': sp.factor, 'expand': sp.expand, 'simplify': sp.simplify,
        'cancel': sp.cancel, 'apart': sp.apart, 'together': sp.together,
        
        # Special functions
        'factorial': sp.factorial, 'binomial': sp.binomial,
        'gamma': sp.gamma, 'erf': sp.erf,
    })


class MathSolver:
    """
    Production-ready mathematical expression solver with comprehensive features
    """
    
    def __init__(self, config: Optional[MathSolverConfig] = None):
        """Initialize the math solver with optional configuration"""
        self.config = config or MathSolverConfig()
        logger.info("MathSolver initialized")
    
    def _validate_input(self, expr: str) -> Tuple[bool, Optional[str]]:
        """
        Validate input expression
        
        Args:
            expr: Input expression string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not expr or not expr.strip():
            return False, "Empty expression provided"
        
        if len(expr) > self.config.MAX_EXPR_LENGTH:
            return False, f"Expression exceeds maximum length of {self.config.MAX_EXPR_LENGTH}"
        
        # Check for potentially dangerous patterns
        dangerous_patterns = ['__', 'import', 'eval', 'exec', 'compile', 'open', 'file']
        for pattern in dangerous_patterns:
            if pattern in expr.lower():
                return False, f"Expression contains forbidden pattern: {pattern}"
        
        return True, None
    
    def _parse_expression(self, expr: str) -> sp.Basic:
        """
        Parse string expression into SymPy expression
        
        Args:
            expr: Mathematical expression as string
            
        Returns:
            SymPy expression object
        """
        try:
            parsed = parse_expr(
                expr,
                transformations=self.config.TRANSFORMATIONS,
                local_dict=self.config.LOCAL_DICT,
                evaluate=False
            )
            return parsed
        except Exception as e:
            logger.error(f"Failed to parse expression '{expr}': {str(e)}")
            raise ValueError(f"Invalid expression syntax: {str(e)}")
    
    def simplify(self, expr: str) -> MathResult:
        """Simplify mathematical expression"""
        is_valid, error = self._validate_input(expr)
        if not is_valid:
            return MathResult(success=False, operation="simplify", input_expr=expr, error=error)
        
        try:
            parsed = self._parse_expression(expr)
            simplified = sp.simplify(parsed)
            numeric = None
            
            try:
                if simplified.is_number or len(simplified.free_symbols) == 0:
                    numeric = str(simplified.evalf())
            except:
                pass
            
            return MathResult(
                success=True,
                operation="simplify",
                input_expr=expr,
                simplified=str(simplified),
                numeric=numeric
            )
        except Exception as e:
            logger.error(f"Simplification failed: {str(e)}")
            return MathResult(success=False, operation="simplify", input_expr=expr, error=str(e))
    
    def solve_equation(self, expr: str) -> MathResult:
        """
        Solve equations or systems of equations
        
        Supports formats:
        - "x^2 - 4 = 0"
        - "x + y = 5 and x - y = 1"
        """
        is_valid, error = self._validate_input(expr)
        if not is_valid:
            return MathResult(success=False, operation="solve", input_expr=expr, error=error)
        
        try:
            # Handle system of equations
            if " and " in expr.lower():
                equations = re.split(r'\s+and\s+', expr, flags=re.IGNORECASE)
                parsed_eqs = []
                symbols_set = set()
                
                for eq in equations:
                    if '=' not in eq:
                        raise ValueError(f"Invalid equation format: {eq}")
                    
                    lhs, rhs = eq.split("=", 1)
                    lhs_parsed = self._parse_expression(lhs.strip())
                    rhs_parsed = self._parse_expression(rhs.strip())
                    
                    parsed_eqs.append(sp.Eq(lhs_parsed, rhs_parsed))
                    symbols_set |= lhs_parsed.free_symbols | rhs_parsed.free_symbols
                
                solution = sp.solve(parsed_eqs, tuple(symbols_set), dict=True)
            else:
                # Single equation
                if '=' in expr:
                    lhs, rhs = expr.split("=", 1)
                    lhs_parsed = self._parse_expression(lhs.strip())
                    rhs_parsed = self._parse_expression(rhs.strip())
                    equation = sp.Eq(lhs_parsed, rhs_parsed)
                else:
                    # Assume equation equals zero
                    equation = self._parse_expression(expr)
                
                solution = sp.solve(equation, dict=True)
            
            return MathResult(
                success=True,
                operation="solve",
                input_expr=expr,
                result=str(solution)
            )
        except Exception as e:
            logger.error(f"Equation solving failed: {str(e)}")
            return MathResult(success=False, operation="solve", input_expr=expr, error=str(e))
    
    def differentiate(self, expr: str, var: str = 'x', order: int = 1) -> MathResult:
        """
        Calculate derivative of expression
        
        Args:
            expr: Expression to differentiate
            var: Variable to differentiate with respect to
            order: Order of derivative
        """
        is_valid, error = self._validate_input(expr)
        if not is_valid:
            return MathResult(success=False, operation="differentiate", input_expr=expr, error=error)
        
        try:
            parsed = self._parse_expression(expr)
            var_symbol = sp.Symbol(var)
            
            derivative = sp.diff(parsed, var_symbol, order)
            simplified = sp.simplify(derivative)
            
            return MathResult(
                success=True,
                operation=f"differentiate (order {order})",
                input_expr=expr,
                result=str(derivative),
                simplified=str(simplified),
                metadata={"variable": var, "order": order}
            )
        except Exception as e:
            logger.error(f"Differentiation failed: {str(e)}")
            return MathResult(success=False, operation="differentiate", input_expr=expr, error=str(e))
    
    def integrate(self, expr: str, var: str = 'x', limits: Optional[Tuple] = None) -> MathResult:
        """
        Calculate integral of expression
        
        Args:
            expr: Expression to integrate
            var: Variable to integrate with respect to
            limits: Optional tuple of (lower, upper) bounds for definite integral
        """
        is_valid, error = self._validate_input(expr)
        if not is_valid:
            return MathResult(success=False, operation="integrate", input_expr=expr, error=error)
        
        try:
            parsed = self._parse_expression(expr)
            var_symbol = sp.Symbol(var)
            
            if limits:
                integral = sp.integrate(parsed, (var_symbol, limits[0], limits[1]))
                op_type = "definite integral"
            else:
                integral = sp.integrate(parsed, var_symbol)
                op_type = "indefinite integral"
            
            simplified = sp.simplify(integral)
            numeric = None
            
            try:
                if simplified.is_number:
                    numeric = str(simplified.evalf())
            except:
                pass
            
            return MathResult(
                success=True,
                operation=op_type,
                input_expr=expr,
                result=str(integral),
                simplified=str(simplified),
                numeric=numeric,
                metadata={"variable": var, "limits": limits}
            )
        except Exception as e:
            logger.error(f"Integration failed: {str(e)}")
            return MathResult(success=False, operation="integrate", input_expr=expr, error=str(e))
    
    def matrix_operations(self, expr: str) -> MathResult:
        """
        Perform matrix operations (determinant, inverse, eigenvalues, etc.)
        
        Example: "Matrix([[1, 2], [3, 4]])"
        """
        is_valid, error = self._validate_input(expr)
        if not is_valid:
            return MathResult(success=False, operation="matrix", input_expr=expr, error=error)
        
        try:
            parsed = self._parse_expression(expr)
            
            if not isinstance(parsed, sp.Matrix):
                return MathResult(
                    success=False,
                    operation="matrix",
                    input_expr=expr,
                    error="Expression is not a valid matrix"
                )
            
            matrix = parsed
            det = matrix.det()
            trace = matrix.trace()
            rank = matrix.rank()
            
            metadata = {
                "shape": f"{matrix.rows}x{matrix.cols}",
                "determinant": str(det),
                "trace": str(trace),
                "rank": str(rank)
            }
            
            # Calculate inverse if square and non-singular
            if matrix.rows == matrix.cols and det != 0:
                metadata["inverse"] = str(matrix.inv())
            else:
                metadata["inverse"] = "N/A (singular or non-square)"
            
            # Calculate eigenvalues for square matrices
            if matrix.rows == matrix.cols:
                try:
                    eigenvals = matrix.eigenvals()
                    metadata["eigenvalues"] = str(eigenvals)
                except:
                    metadata["eigenvalues"] = "Could not compute"
            
            return MathResult(
                success=True,
                operation="matrix",
                input_expr=expr,
                result=str(matrix),
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Matrix operations failed: {str(e)}")
            return MathResult(success=False, operation="matrix", input_expr=expr, error=str(e))
    
    def series_expansion(self, expr: str, var: str = 'x', point: float = 0, order: int = 6) -> MathResult:
        """
        Calculate Taylor/Laurent series expansion
        
        Args:
            expr: Expression to expand
            var: Variable to expand around
            point: Point of expansion
            order: Number of terms
        """
        is_valid, error = self._validate_input(expr)
        if not is_valid:
            return MathResult(success=False, operation="series", input_expr=expr, error=error)
        
        try:
            parsed = self._parse_expression(expr)
            var_symbol = sp.Symbol(var)
            
            series = sp.series(parsed, var_symbol, point, order)
            
            return MathResult(
                success=True,
                operation="series expansion",
                input_expr=expr,
                result=str(series),
                metadata={"variable": var, "point": point, "order": order}
            )
        except Exception as e:
            logger.error(f"Series expansion failed: {str(e)}")
            return MathResult(success=False, operation="series", input_expr=expr, error=str(e))
    
    def process(self, query: str) -> MathResult:
        """
        Main entry point - automatically detect operation type and process
        
        Args:
            query: Mathematical expression or equation
            
        Returns:
            MathResult object
        """
        query = query.strip()
        logger.info(f"Processing query: {query}")
        
        # Detect operation type based on keywords and patterns
        query_lower = query.lower()
        
        # Check for matrix operations
        if query_lower.startswith("matrix"):
            return self.matrix_operations(query)
        
        # Check for equation solving
        if "=" in query:
            return self.solve_equation(query)
        
        # Check for calculus operations with keywords
        if query_lower.startswith("diff") or query_lower.startswith("derivative"):
            # Extract expression after keyword
            match = re.search(r'(?:diff|derivative)\s*\((.+)\)', query, re.IGNORECASE)
            if match:
                return self.differentiate(match.group(1))
        
        if query_lower.startswith("int") or query_lower.startswith("integrate"):
            match = re.search(r'(?:int|integrate)\s*\((.+)\)', query, re.IGNORECASE)
            if match:
                return self.integrate(match.group(1))
        
        # Default to simplification
        return self.simplify(query)


def MathAI(query: str = '1*x + 2*x - 3*x', operation: str = 'auto') -> str:
    """
    Main API function for mathematical operations
    
    Args:
        query: Mathematical expression or equation
        operation: Type of operation ('auto', 'simplify', 'solve', 'differentiate', 'integrate', 'matrix', 'series')
    
    Returns:
        String representation of the result
    
    Examples:
        >>> MathAI('x^2 + 2*x + 1')
        >>> MathAI('x^2 - 4 = 0', operation='solve')
        >>> MathAI('sin(x)', operation='differentiate')
        >>> MathAI('Matrix([[1, 2], [3, 4]])', operation='matrix')
    """
    solver = MathSolver()
    
    try:
        if operation == 'auto':
            result = solver.process(query)
        elif operation == 'simplify':
            result = solver.simplify(query)
        elif operation == 'solve':
            result = solver.solve_equation(query)
        elif operation == 'differentiate':
            result = solver.differentiate(query)
        elif operation == 'integrate':
            result = solver.integrate(query)
        elif operation == 'matrix':
            result = solver.matrix_operations(query)
        elif operation == 'series':
            result = solver.series_expansion(query)
        else:
            return f"Error: Unknown operation type '{operation}'"
        
        return str(result)
    
    except Exception as e:
        logger.error(f"Unexpected error in MathAI: {str(e)}")
        return f"Error: {str(e)}"


# Example usage and tests
if __name__ == "__main__":
    print("=== Mathematical Expression Solver ===\n")
    
    # Test cases
    test_cases = [
        ("1*x + 2*x - 3*x", "auto"),
        ("x^2 - 4 = 0", "solve"),
        ("x + y = 10 and x - y = 2", "solve"),
        ("sin(x)^2 + cos(x)^2", "simplify"),
        ("x^3 + 3*x^2 + 3*x + 1", "simplify"),
        ("Matrix([[1, 2], [3, 4]])", "matrix"),
    ]
    
    for expr, op in test_cases:
        print(f"Query: {expr}")
        print(f"Operation: {op}")
        print(MathAI(expr, op))
        print("-" * 60)
