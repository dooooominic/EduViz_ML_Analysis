# EduViz_ML_Analysis

This is my final project for my Visualization for Machine Learning course at NYU Tandon School of Engineering, Fall 2025. I wanted to work with education data, of course, and I am using this project to practice all that I have learned from 0 CS knowledge this past summer to now, by cleaning up data, applying different machine learning concepts, training different classification and regression models I have learned, and creating visualizations in Streamlit in the form of a data dashboard. There is still a lot more to learn, but this is what I have so far! 

**Overview**

EduViz is an interactive data visualization and machine learning dashboard designed to explore district-level education trends in Texas using statewide assessment data from 2022–2025. The project combines predictive modeling, interpretable ML techniques, and interactive visualizations to help educators, researchers, and policymakers better understand patterns in academic performance and equity over time.

This project was developed as part of a visualization and machine learning research workflow and emphasizes transparency, interpretability, and real-world policy relevance.

EduViz was built with three core goals in mind:
	•	Make complex education data accessible through interactive, intuitive visualizations
	•	Leverage machine learning responsibly to model academic outcomes while avoiding black-box explanations
	•	Support data-informed decision making by surfacing interpretable insights relevant to districts, educators, and stakeholders

Rather than focusing solely on predictive accuracy, EduViz prioritizes understanding why models behave the way they do and how different educational indicators relate to outcomes.

<img width="1406" height="484" alt="image" src="https://github.com/user-attachments/assets/5db171ab-dd60-48e2-9cd8-4925a8db5980" />

<img width="1114" height="528" alt="image" src="https://github.com/user-attachments/assets/f2a5da3a-4630-4fa1-a4a8-0b07258a3ac7" />
<img width="1104" height="529" alt="image" src="https://github.com/user-attachments/assets/1db26ccc-f230-49aa-a968-f26e31e5b373" />

**Tech Stack**

	•	Python (pandas, numpy, scikit-learn, keras)
	
	•	Machine Learning: Logistic Regression, Random Forest, GAMs, Gradient Boosting Regressor, interpretable models
	
	•	Explainability: LIME
	
	•	Visualization: Plotly, Matplotlib
	
	•	Dashboard: Streamlit

**Project Structure**

```
EduViz/
├── __pycache__                     # Cached Python bytecode (auto-generated) 
├── analysis_outputs/                # Saved figures, model results, and evaluation outputs to enhance Streamlit performance
├── data/                            # Processed district-level education datasets
├── scripts/                         # Helper scripts for preprocessing, modeling, and analysis
├── EduViz.py                        # Main Python module for analysis and dashboard logic
├── VizML_Final_Project.ipynb        # Jupyter notebook used for exploratory analysis and model development
├── EduViz_Final_Paper.pdf           # Final IEEE conference style paper describing methods, results, and implications
├── VizML Final Project Presentation.pdf  # Slide deck for project presentation
├── requirements.txt                 # Python dependencies
└── README.md
```

  run in terminal: streamlit run eduviz.py
