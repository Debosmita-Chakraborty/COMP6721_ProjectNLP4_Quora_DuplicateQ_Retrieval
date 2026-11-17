
1. Write the task in one precise sentence and identify the input, output, and success criterion. 

Given a new question and existing questions, the system outputs the top K duplicate questions; it succeeds when duplicates are ranked highly.

2. Name at least one baseline method that your final system must beat or meaningfully compare against.

TF-IDF with cosine similarity and BM25.

3. Choose the primary evaluation metric(s) and state why they match the task. 

Recall@K and MRR, because we want the correct duplicate questions to appear near the top of the results.

4. Define a train/validation/test or experiment protocol that avoids using test results for tuning. 

Use the dev split for tuning and model selection, then use the test split only once for final evaluation.

5. Create/confirm your project folder structure, fixed random seed(s) where applicable, requirements/environment
notes, and a results folder. 

project/
data/
src/
results/
requirements.txt
README.md

6. Write the next two concrete implementation steps for the coming week.

Set up Git and the project environment.
Download and start preparing the dataset.