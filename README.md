# cours_ia_cyber_laval_exploration
Learn how to explore a dataset and machine learning models

# How to

- Fork this repository.
- Clone it locally.
- [Install pixi](https://pixi.prefix.dev/dev/installation/)
- Run `pixi install` to create a clean virtual environment.
- Run `pixi run convert-to-notebooks` if you prefer to work with notebooks rather than python files. 
- If you have too much trouble with pixi, you can use your usual virtual env system, and use the requirements.txt.
- Complete the code.
- Answer the questions below based on your exploration.
- Run `pixi run convert-to-python-files` if you worked in notebooks.
- Push to your fork your code and your answers.

# Steps of the tutorial

Using the notebook "explore_data", get to know the data and answer these questions:
1. How many examples are there in the dataset?
2. What is the distribution of the target?
3. What are the features that can be used to predict the target?
4. Are there any missing values in the dataset? 
5. What is the most common answer to "How much do you personally identify as a Midwesterner"?

Using the notebook "compare models":
6. Among the three models, which one has the best recall?  
7. Among the three models, which one has the best practical application?  
8. Among the three models, which one generalizes the best?  

Once all these are done, you can continue to the [second part of the tutorial](https://github.com/MarieSacksick/midwest_survey_models).


Answers (look at answers.txt to see the expected format):

Question 1: How many examples are there in the dataset?
Number of examples (rows): 2494 
Number of features (columns): 28

Question 2: What is the distribution of the target?
Target distribution:
Census_Region
East North Central    758
West North Central    358
Middle Atlantic       334
South Atlantic        248
Pacific               243
Mountain              190
West South Central    172
East South Central     97
New England            94

Question 3: What are the features that can be used to predict the target?
Columns:
['RespondentID', 'What_would_you_call_the_part_of_the_country_you_live_in_now', 'How_much_do_you_personally_identify_as_a_Midwesterner', 'Do_you_consider_Illinois_state_as_part_of_the_Midwest', 'Do_you_consider_Indiana_state_as_part_of_the_Midwest', 'Do_you_consider_Iowa_state_as_part_of_the_Midwest', 'Do_you_consider_Kansas_state_as_part_of_the_Midwest', 'Do_you_consider_Michigan_state_as_part_of_the_Midwest', 'Do_you_consider_Minnesota_state_as_part_of_the_Midwest', 'Do_you_consider_Missouri_state_as_part_of_the_Midwest', 'Do_you_consider_Nebraska_state_as_part_of_the_Midwest', 'Do_you_consider_North_Dakota_state_as_part_of_the_Midwest', 'Do_you_consider_Ohio_state_as_part_of_the_Midwest', 'Do_you_consider_South_Dakota_state_as_part_of_the_Midwest', 'Do_you_consider_Wisconsin_state_as_part_of_the_Midwest', 'Do_you_consider_Arkansas_state_as_part_of_the_Midwest', 'Do_you_consider_Colorado_state_as_part_of_the_Midwest', 'Do_you_consider_Kentucky_state_as_part_of_the_Midwest', 'Do_you_consider_Oklahoma_state_as_part_of_the_Midwest', 'Do_you_consider_Pennsylvania_state_as_part_of_the_Midwest', 'Do_you_consider_West_Virginia_state_as_part_of_the_Midwest', 'Do_you_consider_Montana_state_as_part_of_the_Midwest', 'Do_you_consider_Wyoming_state_as_part_of_the_Midwest', 'Gender', 'Age', 'Household_Income', 'Education', 'In_what_ZIP_code_is_your_home_located']

# How many features are numerical? How many are categorical (text)?
Data types:
RespondentID                                                   float64
What_would_you_call_the_part_of_the_country_you_live_in_now        str
How_much_do_you_personally_identify_as_a_Midwesterner              str
Do_you_consider_Illinois_state_as_part_of_the_Midwest              str
Do_you_consider_Indiana_state_as_part_of_the_Midwest               str
Do_you_consider_Iowa_state_as_part_of_the_Midwest                  str
Do_you_consider_Kansas_state_as_part_of_the_Midwest                str
Do_you_consider_Michigan_state_as_part_of_the_Midwest              str
Do_you_consider_Minnesota_state_as_part_of_the_Midwest             str
Do_you_consider_Missouri_state_as_part_of_the_Midwest              str
Do_you_consider_Nebraska_state_as_part_of_the_Midwest              str
Do_you_consider_North_Dakota_state_as_part_of_the_Midwest          str
Do_you_consider_Ohio_state_as_part_of_the_Midwest                  str
Do_you_consider_South_Dakota_state_as_part_of_the_Midwest          str
Do_you_consider_Wisconsin_state_as_part_of_the_Midwest             str
Do_you_consider_Arkansas_state_as_part_of_the_Midwest              str
Do_you_consider_Colorado_state_as_part_of_the_Midwest              str
Do_you_consider_Kentucky_state_as_part_of_the_Midwest              str
Do_you_consider_Oklahoma_state_as_part_of_the_Midwest              str
Do_you_consider_Pennsylvania_state_as_part_of_the_Midwest          str
Do_you_consider_West_Virginia_state_as_part_of_the_Midwest         str
Do_you_consider_Montana_state_as_part_of_the_Midwest               str
Do_you_consider_Wyoming_state_as_part_of_the_Midwest               str
Gender                                                             str
Age                                                                str
Household_Income                                                   str
Education                                                          str
In_what_ZIP_code_is_your_home_located                              str
dtype: object

One feature is numerical and the others are categorical.


Question 4: Are there any missing values in the dataset?
Columns with missing values:
Series([], dtype: int64)


Question 5: What is the most common answer to "How much do you personally identify as a Midwesterner"?
Value counts for 'How_much_do_you_personally_identify_as_a_Midwesterner':
How_much_do_you_personally_identify_as_a_Midwesterner
Not at all    965
A lot         697
Some          528
Not much      304

The most common answer is: "Not at all"

Question 6: Among the three models, which one has the best recall?
Of the three models, the one with the best recall for the positive class "North Central" is Gradient Boosting.

Question 7: Which model has the best practical application?
The model with the best practical application is Gradient Boosting, as it achieves the highest practical score (4629), and maximizing useful predictions while limiting costly errors.

Question 8: Which model generalizes the best?
The model that generalizes best is Random Forest: its test accuracy (0.944) is close to its average cross-validation accuracy (≈0.935), with a reasonable difference between training and test, which indicates a good generalization capability.