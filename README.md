CAT-LLM: Prompting Large Language Models with Text Style Definition for Chinese Article-style Transfer
====
Project Structure<br>
----
The project is organized into several key directories and modules. Here's an overview of the project structure:
.
├── bert-base-chinese      # Store bert-base-chines file used in our experiment, .<br>
├── assets        # Store project assets, such as images, diagrams, or any visual elements used to enhance the presentation and understanding of the project.<br>
├── configs       # Store configuration files.<br>
├── core          # Core codebase.<br>
│   ├── data      # Data processing module.<br>
│   ├── evaluator # Evaluator module.<br>
│   └── llm       # Load Large Language Models (LLMs) module.<br>
├── data          # Store datasets and data processing scripts.<br>
├── external      # Store the Grimoire Ranking model based on the classifier approach.<br>
├── outputs       # Store experiment output files.<br>
├── prompts       # Store text files used as prompts when interacting with LLMs.<br>
├── stats         # Store experiment statistical results.<br>
└── tests         # Store test code or unit tests.<br>
