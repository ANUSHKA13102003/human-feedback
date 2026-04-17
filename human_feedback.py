# Human Feedback Annotation System

## Overview
This script implements a basic human annotation system for scoring responses based on various criteria. It allows human annotators to score responses, and save their scores for analysis.

## Features
- Input responses for evaluation
- Score responses based on predefined criteria
- Save scores to a CSV file

## Example Usage:
```python
import csv

class HumanFeedback:
    def __init__(self, criteria):
        self.criteria = criteria
        self.scores = []

    def score_response(self, response):
        print(f"Scoring response: {response}")
        score = {criterion: input(f'Rate {criterion} (1-5): ') for criterion in self.criteria}
        self.scores.append({"response": response, **score})

    def save_scores(self, filename='scores.csv'):
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.scores[0].keys())
            writer.writeheader()
            writer.writerows(self.scores)

# Usage
criteria = ['Relevance', 'Clarity', 'Engagement']
feedback_system = HumanFeedback(criteria)
feedback_system.score_response('This is a sample response.')
feedback_system.save_scores() 
```

## Requirements
- Python 3.x
- CSV library (included in Python standard libraries)  
