
## Running the application

In the project root (streamlit_franke) type in the terminal and run : streamlit run dashboard.py

## Want to add new models?

check the models directory
append the model in the model_trainer.py (and also in the model manager file) file

go to the "input" directory, open the input_handler.py
1 Add the model's name into the model_options for the get_user_input() function

#### NB: values in the "allowed_models" variable (mentioned below) must match what is in the "model_options" inside the get_user_input() function

go to the dashboard.py file
Add the model's configurations as a key-value (dictionary) inside the config variable (config is located at the top of the dashboard.py) 
you can add the new model in the "allowed_models" variable (check NB above)


## Want to add a new metric?
in the models directory add the metric in the metric_calculator.py file

currently, the metric is based on the lowest SMAPE value from all models. Feel free to change it in the model_selector.py file

## Proeject Structure
project was initially with this structure, but changed. It may not remain like this (due to future optimizations, or new directory and files generated during running of the application etc).

streamlit_franke/
├── dashboard.py
├── README.md
├── data_prep/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── data_transformer.py
│   ├── holiday_utils.py
│   └── utils.py
├── models/
│   ├── __init__.py
│   ├── model_trainer.py
│   ├── model_saver.py
│   ├── model_selector.py
│   ├── forecast_generator.py
│   └── utils.py
├── eval/
│   ├── __init__.py
│   ├── metric_calculator.py
│   ├── evaluation_report.py
│   └── utils.py
├── input/
│   ├── __init__.py
│   ├── input_handler.py
│   ├── data_preview.py
│   └── utils.py
├── reports/
│   ├── __init__.py
│   ├── visualization.py
│   ├── report_generator.py
│   └── utils.py
└── utils/
    ├── __init__.py
    ├── date_utils.py
    ├── config_loader.py
    ├── streamlit_utils.py
    └── general_utils.py