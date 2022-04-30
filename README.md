# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity
- Author: Leonardo Meireles Murtha Oliveira

## Project Description

This project aims to cover best practices when developing a ML project.

The ML itself targets predicting curstomer churn using a bank data.

It covers:

* Best coding practices: code modularization, code refactoring
* PEP8 standard
* Testing/Automation with pytest
* Logging

## Files and data description

**Folders:**

> data: contains all the data necessary to run the project;
>
> images: contains images generated in EDA, Model Evaluation;
>
> logs: contain logs of tests and the churn library;
>
> models: contains the models generated by running the the project;
>
> mock: contains mocked data, models necessary to run the automated tests.

**Files (Important Scripts):**

> churn_library.py : contains all helper functions to predict customer churn plus EDA of the dataset and model evaluation;
>
> churn_script_logging_and_tests.py : pytest script to test all helper functions from churn_library script (also everything is being logged).

## Running Files

How do you run your files? What should happen when you run your files?

First install requirements with requirements_version.txt files provided.

---

> **To run the project:**
>
> python3 churn_library.py
>
> *It runs the entire pipeline creating EDA images/reports, preprocessing data, feature engineering and training the models which are saved in the /models folder.*
>
> ---
>
> **To test the project:**
>
> pytest churn_script_logging_and_tests.pychurn_library.py
>
> *It runs the automated tests for the helper functions of the pipelinel, everything test should pass otherwise the project is broken.*
