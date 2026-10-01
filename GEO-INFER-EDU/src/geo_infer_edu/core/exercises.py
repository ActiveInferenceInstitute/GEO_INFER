"""
Interactive exercise generation module.

Provides exercise creation for geospatial education including spatial analysis,
coding exercises, problem-based learning scenarios, and assessments.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
from enum import Enum
import random
import hashlib

logger = logging.getLogger(__name__)


class ExerciseType(Enum):
    """Types of educational exercises."""

    MAPPING = "mapping"
    ANALYSIS = "analysis"
    CODING = "coding"
    PROBLEM_SOLVING = "problem_solving"
    MULTIPLE_CHOICE = "multiple_choice"
    PRACTICAL = "practical"
    PROJECT = "project"


class DifficultyLevel(Enum):
    """Exercise difficulty levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class Exercise:
    """Represents an educational exercise."""

    id: str
    title: str
    description: str
    exercise_type: ExerciseType
    difficulty: DifficultyLevel
    concepts: list[str]
    instructions: str
    expected_duration_minutes: int
    hints: list[str] = field(default_factory=list)
    starter_code: str | None = None
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    rubric: dict[str, Any] = field(default_factory=dict)
    resources: list[str] = field(default_factory=list)


@dataclass
class Assessment:
    """Represents an assessment with multiple items."""

    id: str
    title: str
    description: str
    items: list[Exercise]
    time_limit_minutes: int | None = None
    passing_score: float = 0.7
    rubrics: dict[str, dict] = field(default_factory=dict)


class ExerciseGenerator:
    """
    Generate interactive educational exercises for geospatial learning.

    Supports various exercise formats including mapping, analysis, coding,
    problem-based learning, and assessments with automatic feedback.
    """

    def __init__(
        self,
        exercise_types: list[str] | None = None,
        difficulty_scaling: str = "adaptive",
        feedback_mode: str = "immediate",
    ):
        """
        Initialize exercise generator.

        Args:
            exercise_types: Types of exercises to generate
            difficulty_scaling: How difficulty adjusts ('adaptive', 'fixed', 'progressive')
            feedback_mode: When feedback is provided ('immediate', 'delayed', 'on_submit')
        """
        self.exercise_types = exercise_types or ["mapping", "analysis", "coding"]
        self.difficulty_scaling = difficulty_scaling
        self.feedback_mode = feedback_mode
        self._exercise_counter = 0
        logger.info(f"Initialized ExerciseGenerator with types: {self.exercise_types}")

    def _generate_id(self, prefix: str = "ex") -> str:
        """Generate unique exercise ID."""
        self._exercise_counter += 1
        hash_input = f"{prefix}{self._exercise_counter}{random.random()}"
        return f"{prefix}_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"

    def create(
        self,
        concepts: list[str],
        format: str = "interactive_map",
        difficulty: str = "progressive",
        include_hints: bool = True,
    ) -> list[Exercise]:
        """
        Create exercises for specified concepts.

        Args:
            concepts: Concepts to cover in exercises
            format: Exercise format ('interactive_map', 'code', 'quiz', 'practical')
            difficulty: Difficulty setting ('beginner', 'progressive', 'advanced')
            include_hints: Whether to include hints

        Returns:
            List of Exercise objects
        """
        exercises = []

        # Determine difficulty progression
        if difficulty == "progressive":
            difficulties = [
                DifficultyLevel.BEGINNER,
                DifficultyLevel.INTERMEDIATE,
                DifficultyLevel.ADVANCED,
            ]
        else:
            difficulties = [DifficultyLevel(difficulty)] * len(concepts)

        for i, concept in enumerate(concepts):
            diff_level = difficulties[i % len(difficulties)]

            exercise = self._create_concept_exercise(
                concept=concept,
                format=format,
                difficulty=diff_level,
                include_hints=include_hints,
            )
            exercises.append(exercise)

        logger.info(f"Created {len(exercises)} exercises for {len(concepts)} concepts")
        return exercises

    def _create_concept_exercise(
        self,
        concept: str,
        format: str,
        difficulty: DifficultyLevel,
        include_hints: bool,
    ) -> Exercise:
        """Create a single exercise for a concept."""
        concept_display = concept.replace("_", " ").title()

        # Map format to exercise type
        format_type_map = {
            "interactive_map": ExerciseType.MAPPING,
            "code": ExerciseType.CODING,
            "quiz": ExerciseType.MULTIPLE_CHOICE,
            "practical": ExerciseType.PRACTICAL,
        }
        exercise_type = format_type_map.get(format, ExerciseType.ANALYSIS)

        # Generate exercise content
        exercise = Exercise(
            id=self._generate_id("ex"),
            title=f"{concept_display} Exercise",
            description=f"Practice {concept_display} skills through hands-on activities.",
            exercise_type=exercise_type,
            difficulty=difficulty,
            concepts=[concept],
            instructions=self._generate_instructions(
                concept, exercise_type, difficulty
            ),
            expected_duration_minutes=self._estimate_duration(difficulty),
            hints=self._generate_hints(concept, difficulty) if include_hints else [],
            resources=[f"GEO-INFER documentation on {concept}"],
        )

        # Add type-specific content
        if exercise_type == ExerciseType.CODING:
            exercise.starter_code = self._generate_starter_code(concept)
            exercise.test_cases = self._generate_test_cases(concept)

        return exercise

    def _generate_instructions(
        self, concept: str, exercise_type: ExerciseType, difficulty: DifficultyLevel
    ) -> str:
        """Generate exercise instructions."""
        concept_display = concept.replace("_", " ")

        instructions_templates = {
            ExerciseType.MAPPING: f"""
In this exercise, you will apply {concept_display} techniques to analyze spatial data.

**Objectives:**
1. Load the provided dataset
2. Apply {concept_display} analysis
3. Visualize the results on an interactive map
4. Interpret the findings

**Deliverables:**
- Interactive map showing analysis results
- Brief written interpretation of findings
""",
            ExerciseType.CODING: f"""
In this coding exercise, you will implement {concept_display} algorithms.

**Objectives:**
1. Complete the provided function skeleton
2. Handle edge cases appropriately
3. Ensure all test cases pass
4. Optimize for performance

**Requirements:**
- Follow PEP 8 style guidelines
- Include docstrings
- Handle exceptions gracefully
""",
            ExerciseType.ANALYSIS: f"""
Analyze the provided spatial data using {concept_display} methods.

**Steps:**
1. Explore and understand the dataset
2. Formulate analytical questions
3. Apply appropriate {concept_display} techniques
4. Document your findings
""",
            ExerciseType.PRACTICAL: f"""
Apply {concept_display} to solve a real-world problem.

**Scenario:**
You are tasked with analyzing spatial patterns using {concept_display}.

**Tasks:**
1. Review the problem context
2. Select appropriate methods
3. Execute the analysis
4. Present your findings
""",
        }

        return instructions_templates.get(
            exercise_type,
            f"Complete the {concept_display} exercise following the provided guidelines.",
        )

    def _estimate_duration(self, difficulty: DifficultyLevel) -> int:
        """Estimate exercise duration based on difficulty."""
        duration_map = {
            DifficultyLevel.BEGINNER: 20,
            DifficultyLevel.INTERMEDIATE: 40,
            DifficultyLevel.ADVANCED: 60,
            DifficultyLevel.EXPERT: 90,
        }
        return duration_map.get(difficulty, 30)

    def _generate_hints(self, concept: str, difficulty: DifficultyLevel) -> list[str]:
        """Generate hints for an exercise."""
        concept_display = concept.replace("_", " ")

        base_hints = [
            f"Review the documentation on {concept_display}",
            f"Consider the fundamental principles of {concept_display}",
            "Break the problem into smaller steps",
        ]

        if difficulty == DifficultyLevel.BEGINNER:
            base_hints.extend(
                [
                    "Start with the simplest case first",
                    "Use the example code as a template",
                ]
            )
        elif difficulty == DifficultyLevel.ADVANCED:
            base_hints = [base_hints[0]]  # Fewer hints for advanced

        return base_hints

    def _generate_starter_code(self, concept: str) -> str:
        """Generate starter code for coding exercises."""
        concept_snake = concept.lower().replace(" ", "_")

        return f'''"""
{concept.replace("_", " ").title()} Exercise

Complete the functions below to implement {concept.replace("_", " ")} analysis.
"""

def analyze_{concept_snake}(data, parameters=None):
    """
    Analyze spatial data using {concept.replace("_", " ")} techniques.
    
    Args:
        data: Input spatial data
        parameters: Optional analysis parameters
        
    Returns:
        Analysis results
    """
    if data is None:
        return {{
            "concept": "{concept}",
            "status": "empty_input",
            "parameters": parameters or {{}},
        }}

    results = {{
        "concept": "{concept}",
        "status": "ready_for_extension",
        "input_type": type(data).__name__,
        "parameters": parameters or {{}},
    }}
    
    return results


def visualize_results(results, output_path=None):
    """
    Visualize the analysis results.
    
    Args:
        results: Analysis results to visualize
        output_path: Optional path to save visualization
        
    Returns:
        Visualization object
    """
    visualization = {{
        "results": results,
        "output_path": output_path,
        "rendered": results is not None,
    }}
    return visualization


if __name__ == "__main__":
    # Example usage
    sample_data = None  # Load your data here
    results = analyze_{concept_snake}(sample_data)
    visualize_results(results)
'''

    def _generate_test_cases(self, concept: str) -> list[dict[str, Any]]:
        """Generate test cases for coding exercises."""
        return [
            {
                "name": "test_basic_functionality",
                "input": {"data": "sample_data", "parameters": None},
                "expected": "not_none",
                "description": f"Test basic {concept} functionality",
            },
            {
                "name": "test_empty_input",
                "input": {"data": None, "parameters": None},
                "expected": "handles_gracefully",
                "description": "Test handling of empty input",
            },
            {
                "name": "test_with_parameters",
                "input": {"data": "sample_data", "parameters": {"threshold": 0.5}},
                "expected": "not_none",
                "description": "Test with custom parameters",
            },
        ]

    def create_coding_exercises(
        self,
        topic: str,
        language: str = "python",
        framework: str = "geo_infer",
        test_cases: bool = True,
        starter_code: bool = True,
    ) -> list[Exercise]:
        """
        Generate coding exercises for a topic.

        Args:
            topic: Topic for coding exercises
            language: Programming language
            framework: Framework to use
            test_cases: Whether to include test cases
            starter_code: Whether to include starter code

        Returns:
            List of coding Exercise objects
        """
        exercises = []

        # Generate exercises with increasing difficulty
        difficulty_concepts = [
            (DifficultyLevel.BEGINNER, f"basic_{topic}"),
            (DifficultyLevel.INTERMEDIATE, f"intermediate_{topic}"),
            (DifficultyLevel.ADVANCED, f"advanced_{topic}"),
        ]

        for difficulty, concept in difficulty_concepts:
            exercise = Exercise(
                id=self._generate_id("code"),
                title=f"{concept.replace('_', ' ').title()} Coding Exercise",
                description=f"Implement {topic} functionality using {framework}.",
                exercise_type=ExerciseType.CODING,
                difficulty=difficulty,
                concepts=[topic],
                instructions=self._generate_instructions(
                    topic, ExerciseType.CODING, difficulty
                ),
                expected_duration_minutes=self._estimate_duration(difficulty),
                starter_code=(
                    self._generate_starter_code(topic) if starter_code else None
                ),
                test_cases=self._generate_test_cases(topic) if test_cases else [],
                hints=self._generate_hints(topic, difficulty),
            )
            exercises.append(exercise)

        logger.info(f"Created {len(exercises)} coding exercises for {topic}")
        return exercises

    def create_pbl_scenario(
        self,
        context: str,
        problem: str,
        data_provided: list[str],
        expected_deliverables: list[str],
    ) -> Exercise:
        """
        Create a problem-based learning scenario.

        Args:
            context: Real-world context (e.g., 'urban_planning')
            problem: Problem to solve (e.g., 'optimize_fire_station_locations')
            data_provided: List of datasets provided
            expected_deliverables: Expected outputs from learners

        Returns:
            PBL Exercise object
        """
        problem_title = problem.replace("_", " ").title()
        context_title = context.replace("_", " ").title()

        instructions = f"""
# Problem-Based Learning Scenario: {problem_title}

## Context
You are working as a GIS analyst in the {context_title} domain. 
You have been tasked with solving a real-world problem using geospatial analysis.

## Problem Statement
{problem_title}

## Available Data
The following datasets are provided for your analysis:
{chr(10).join(f"- {d}" for d in data_provided)}

## Deliverables
You are expected to produce:
{chr(10).join(f"{i + 1}. {d}" for i, d in enumerate(expected_deliverables))}

## Evaluation Criteria
- Technical accuracy of analysis
- Appropriateness of methods selected
- Quality of visualization and presentation
- Clarity of written justification
- Creativity and innovation in approach
"""

        exercise = Exercise(
            id=self._generate_id("pbl"),
            title=f"PBL: {problem_title}",
            description=f"Problem-based learning scenario in {context_title}",
            exercise_type=ExerciseType.PROJECT,
            difficulty=DifficultyLevel.ADVANCED,
            concepts=[context, problem],
            instructions=instructions,
            expected_duration_minutes=180,  # 3 hours for PBL
            resources=data_provided,
            rubric={
                "technical_accuracy": {
                    "weight": 0.3,
                    "description": "Correctness of analysis",
                },
                "method_selection": {
                    "weight": 0.2,
                    "description": "Appropriate methods chosen",
                },
                "visualization": {
                    "weight": 0.2,
                    "description": "Quality of visualizations",
                },
                "documentation": {
                    "weight": 0.2,
                    "description": "Clarity of written work",
                },
                "innovation": {"weight": 0.1, "description": "Creative approaches"},
            },
        )

        logger.info(f"Created PBL scenario: {problem_title}")
        return exercise

    def create_assessment(
        self,
        learning_objectives: list[dict[str, str]],
        item_types: list[str],
        difficulty_distribution: dict[str, float],
        rubrics: bool = True,
    ) -> Assessment:
        """
        Create an assessment with multiple items.

        Args:
            learning_objectives: Objectives to assess
            item_types: Types of items ('multiple_choice', 'practical', 'project')
            difficulty_distribution: Distribution of difficulties (e.g., {'easy': 0.3, 'medium': 0.5, 'hard': 0.2})
            rubrics: Whether to generate rubrics

        Returns:
            Assessment object
        """
        items = []

        # Generate items for each objective
        for i, objective in enumerate(learning_objectives):
            # Determine difficulty based on distribution
            rand = random.random()
            cumulative = 0.0
            difficulty = DifficultyLevel.INTERMEDIATE

            difficulty_map = {
                "easy": DifficultyLevel.BEGINNER,
                "medium": DifficultyLevel.INTERMEDIATE,
                "hard": DifficultyLevel.ADVANCED,
            }

            for diff_name, prob in difficulty_distribution.items():
                cumulative += prob
                if rand < cumulative:
                    difficulty = difficulty_map.get(
                        diff_name, DifficultyLevel.INTERMEDIATE
                    )
                    break

            # Select item type
            item_type = item_types[i % len(item_types)]
            exercise_type = {
                "multiple_choice": ExerciseType.MULTIPLE_CHOICE,
                "practical": ExerciseType.PRACTICAL,
                "project": ExerciseType.PROJECT,
            }.get(item_type, ExerciseType.PRACTICAL)

            item = Exercise(
                id=self._generate_id("assess"),
                title=f"Assessment Item {i + 1}",
                description=objective.get("description", f"Assess objective {i + 1}"),
                exercise_type=exercise_type,
                difficulty=difficulty,
                concepts=[objective.get("concept", "general")],
                instructions=f"Demonstrate mastery of: {objective.get('description', 'objective')}",
                expected_duration_minutes=15 if item_type == "multiple_choice" else 30,
            )
            items.append(item)

        assessment = Assessment(
            id=self._generate_id("test"),
            title="Competency Assessment",
            description="Assessment of learning objectives",
            items=items,
            time_limit_minutes=sum(item.expected_duration_minutes for item in items),
            passing_score=0.7,
            rubrics=self._generate_assessment_rubrics(items) if rubrics else {},
        )

        logger.info(f"Created assessment with {len(items)} items")
        return assessment

    def _generate_assessment_rubrics(self, items: list[Exercise]) -> dict[str, dict]:
        """Generate rubrics for assessment items."""
        rubrics = {}
        for item in items:
            rubrics[item.id] = {
                "criteria": [
                    {
                        "name": "Understanding",
                        "weight": 0.3,
                        "levels": ["Emerging", "Developing", "Proficient", "Exemplary"],
                    },
                    {
                        "name": "Application",
                        "weight": 0.4,
                        "levels": ["Emerging", "Developing", "Proficient", "Exemplary"],
                    },
                    {
                        "name": "Communication",
                        "weight": 0.3,
                        "levels": ["Emerging", "Developing", "Proficient", "Exemplary"],
                    },
                ]
            }
        return rubrics
