# **OptiMUS**: Optimization Modeling Using mip Solvers and large language models


This repository contains the official implementation for the following three papers (you can use branches to access the other versions):


- V0.1: [OptiMUS: Optimization Modeling Using mip Solvers and large language models](https://arxiv.org/pdf/2310.06116).

- V0.2: [OptiMUS: Scalable Optimization Modeling with (MI) LP Solvers and Large Language Models](https://arxiv.org/pdf/2402.10172).

- V0.3: [OptiMUS-0.3: Using Large Language Models to Model and Solve Optimization Problems at Scale](https://arxiv.org/abs/2407.19633)

<img width="845" alt="image" src="https://github.com/user-attachments/assets/72fbbf20-32ee-4715-a2d5-819133a346ee">



#### Live demo: https://optimus-solver.com/

## NLP4LP Dataset

You can download the dataset from [https://huggingface.co/datasets/udell-lab/NLP4LP](https://huggingface.co/datasets/udell-lab/NLP4LP). Please note that NLP4LP is intended and licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

#### References

**OptiMUS** has two available implementations

**OptiMUS v1** adopts a sequential work-flow implementation. Suitable for small and medium-sized problems.

```
@article{ahmaditeshnizi2023optimus,
  title={OptiMUS: Optimization Modeling Using mip Solvers and large language models},
  author={AhmadiTeshnizi, Ali and Gao, Wenzhi and Udell, Madeleine},
  journal={arXiv preprint arXiv:2310.06116},
  year={2023}
}
```

**OptiMUS v2** adopts agent-based implementation. Suitable for large and complicated tasks.

```
@article{ahmaditeshnizi2024optimus,
  title={OptiMUS: Scalable Optimization Modeling with (MI) LP Solvers and Large Language Models},
  author={AhmadiTeshnizi, Ali and Gao, Wenzhi and Udell, Madeleine},
  journal={arXiv preprint arXiv:2402.10172},
  year={2024}
}
```

**OptiMUS v3** adds RAG and large-scale optimization techniques. 

```
@article{ahmaditeshnizi2024optimus,
  title={OptiMUS-0.3: Using Large Language Models to Model and Solve Optimization Problems at Scale},
  author={AhmadiTeshnizi, Ali and Gao, Wenzhi and Brunborg, Herman and Talaei, Shayan and Udell, Madeleine},
  journal={arXiv preprint arXiv:2407.19633},
  year={2024}
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=teshnizi/OptiMUS&type=date&legend=top-left)](https://www.star-history.com/#teshnizi/OptiMUS&type=date&legend=top-left)

## Workflow Overview

This section explains how the OptiMUS system works and how all the files connect together.

### Problem Input

Problems are provided as directories containing:
- **`desc.txt`**: Natural language description of the optimization problem
- **`params.json`**: JSON file defining parameters (known values) with their shapes, types, and actual values
- **`labels.json`** (optional): Problem categorization labels for RAG-based retrieval

**Example Problem Structure:**
```
problem_directory/
├── desc.txt          # Problem description in natural language
├── params.json       # Parameter definitions and values
└── labels.json       # Problem categories (optional, for RAG)
```

**Example `desc.txt`:**
```
A company produces two types of products: A and B. 
Product A requires 2 hours of labor and 3 units of material.
Product B requires 4 hours of labor and 1 unit of material.
The company has 100 hours of labor and 60 units of material available.
Product A yields a profit of $5 per unit, and Product B yields $3 per unit.
How many units of each product should be produced to maximize profit?
```

**Example `params.json`:**
```json
{
  "LaborHours": {
    "shape": [],
    "type": "int",
    "definition": "Total available labor hours",
    "value": 100
  },
  "MaterialUnits": {
    "shape": [],
    "type": "int", 
    "definition": "Total available material units",
    "value": 60
  }
}
```

### Execution Flow

The system follows a sequential pipeline executed by `main.py`:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION (utils.py: create_state)                  │
│    - Reads desc.txt and params.json from problem directory   │
│    - Extracts parameter values → saves to data.json          │
│    - Creates initial state with description and parameters    │
│    - Saves: state_1_params.json                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. EXTRACT PARAMETERS (parameters.py: get_params)            │
│    - LLM identifies all parameters from description           │
│    - Validates parameters (checks if values are known)        │
│    - Output: Dictionary of parameters with shapes/types       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. EXTRACT OBJECTIVE (objective.py: get_objective)           │
│    - LLM extracts optimization goal from description          │
│    - Optional: RAG retrieves similar problem objectives      │
│    - Output: Objective description                           │
│    - Saves: state_2_objective.json                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. EXTRACT CONSTRAINTS (constraint.py: get_constraints)      │
│    - LLM identifies all constraints from description          │
│    - Optional: RAG retrieves similar problem constraints      │
│    - Validates constraints (removes redundant ones)           │
│    - Output: List of constraint descriptions                 │
│    - Saves: state_3_constraints.json                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. MODEL CONSTRAINTS (constraint_model.py)                  │
│    - LLM converts each constraint to mathematical formulation │
│    - Creates variables needed for constraints                │
│    - Generates auxiliary constraints if needed                │
│    - Optional: RAG retrieves similar constraint formulations │
│    - Output: LaTeX formulations + variable definitions        │
│    - Saves: state_4_constraints_modeled.json                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. MODEL OBJECTIVE (objective_model.py)                     │
│    - LLM converts objective to mathematical formulation       │
│    - Optional: RAG retrieves similar objective formulations  │
│    - Output: LaTeX formulation of objective                  │
│    - Saves: state_5_objective_modeled.json                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. GENERATE CODE (target_code.py: get_codes)                 │
│    - LLM converts LaTeX formulations to Gurobi Python code    │
│    - Generates code for each constraint                      │
│    - Generates code for objective function                   │
│    - Output: Python code snippets for constraints/objective   │
│    - Saves: state_6_code.json                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. ASSEMBLE CODE (generate_code.py: generate_code)           │
│    - Combines all code snippets into complete Python file    │
│    - Adds imports, model setup, parameter loading            │
│    - Adds variable definitions, constraints, objective        │
│    - Adds optimization call and output handling              │
│    - Output: code.py (complete runnable Gurobi script)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 9. EXECUTE & DEBUG (execute_code.py: execute_and_debug)      │
│    - Runs the generated code.py                              │
│    - If errors occur: LLM debugs and fixes the code           │
│    - Iterates up to max_tries times until success            │
│    - Output: Optimal solution or error logs                   │
└─────────────────────────────────────────────────────────────┘
```

### Key Files and Their Roles

- **`main.py`**: Orchestrates the entire pipeline, loads/saves state between steps
- **`parameters.py`**: Extracts and validates parameters from problem description
- **`objective.py`**: Extracts the optimization objective (maximize/minimize goal)
- **`constraint.py`**: Extracts and validates constraints from problem description
- **`constraint_model.py`**: Converts constraint descriptions to mathematical formulations (LaTeX)
- **`objective_model.py`**: Converts objective description to mathematical formulation (LaTeX)
- **`target_code.py`**: Converts mathematical formulations to Gurobi Python code
- **`generate_code.py`**: Assembles all code snippets into a complete runnable script
- **`execute_code.py`**: Executes the generated code and debugs errors using LLM
- **`utils.py`**: Utility functions (state management, LLM calls, JSON extraction)
- **`rag/query_vector_db.py`**: Retrieves similar problems/constraints/objectives from vector databases for RAG

### State Management

The system uses JSON state files to save progress at each step:
- `state_1_params.json`: Parameters extracted
- `state_2_objective.json`: Objective extracted
- `state_3_constraints.json`: Constraints extracted
- `state_4_constraints_modeled.json`: Constraint formulations + variables
- `state_5_objective_modeled.json`: Objective formulation
- `state_6_code.json`: Code snippets generated

This allows resuming from any step if the process is interrupted.

### RAG (Retrieval-Augmented Generation)

When RAG mode is enabled (`--rag-mode`), the system:
- Searches vector databases for similar problems/constraints/objectives
- Uses these examples to improve LLM prompts
- Three RAG modes available:
  - `PROBLEM_DESCRIPTION`: Retrieves similar problems by description
  - `CONSTRAINT_OR_OBJECTIVE`: Retrieves similar constraints/objectives individually
  - `PROBLEM_LABELS`: Retrieves problems by category labels

### Running a Problem

```bash
python main.py --dir /path/to/problem_directory --devmode 1 --rag-mode problem_description
```

The system will:
1. Read the problem from the directory
2. Process it through all pipeline steps
3. Generate and execute Gurobi code
4. Save all intermediate states and final solution in a `run_*` directory


