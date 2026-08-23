Summary of MLOps Practitioner Course Community Messages

Course Overview

· Course: The MLOps Practitioner
· Platform: ZOMRA (https://zomra.io/courses/the-mlops-practitioner)
· Schedule: Weekly sessions on Sundays at 7 PM (first session was last Sunday)
· Duration: ~3 hours per session
· Recordings: Available for 48 hours after each session (first session remains available longer)

Key Resources

1. Course Materials & Projects

· All course materials, mini projects, and final projects are on ZOMRA
· Mini Projects Guide + Final Projects Document:
  · Contains ALL mini projects for each session
  · 2 final projects with details
  · LLM Project dataset
  · Additional resources
  · Deliverables required
  · Prerequisites
· Link: https://drive.google.com/file/d/1NTXz0iNCPitlRezNLLqeDms8SVUZ8MGW/view?usp=sharing

2. Website

· Beta Version: mlopsmena.com
· All information available there before asking questions

3. YouTube Channel

· Playlist: "The MLOps Practitioner Course"
· All sessions will be uploaded to YouTube

Project Information

Mini Projects

· Not mandatory but highly recommended for practical learning
· Each session has associated mini projects
· Deliverables specified in the guide document

Final Projects

· Not mandatory
· Certificate difference:
  · Attendance certificate: Just fill attendance forms for 5 sessions
  · Completed certificate: Complete Final Project

Key Technical Questions & Answers

Q: Is DictVectorizer like OneHotEncoding?

A: Similar but different:

· DictVectorizer automatically does one-hot encoding for categorical/string values
· Leaves numeric values unchanged
· Takes list of dictionaries and understands the difference between types automatically
· OneHotEncoder requires ColumnTransformer to specify categorical columns separately

Q: Can I choose a different model than mentioned in the project?

A: Yes, any model works. DictVectorizer is just preprocessing. However, it's recommended to stick with simple ML models for ease of application.

Q: Are Mini Projects mandatory?

A: No, not mandatory. But without practice, learning is incomplete. The process requires both theoretical and practical work.

Additional Learning Resources

Docker Session

· Extra session about Docker on Saturday at 7 PM
· Provided by DevOps Engineer
· Recording will be uploaded to YouTube

Linux Resources

· Community provides Linux learning resources
· Post: https://www.linkedin.com/posts/mlops-mena_%D9%87%D9%88-%D8%A3%D9%86%D8%A7-%D9%83%D9%80ml-engineer-%D9%85%D8%AD%D8%AA%D8%A7%D8%AC-linux-%D9%81%D8%B9%D9%84%D8%A7-activity-7495960419791048704-wZGA

Backend Fundamentals

· Recommended playlist: Hussein Nasser's YouTube channel
· Focus on: Backend fundamentals, OOP concepts
· Playlist: https://youtube.com/playlist?list=PLui3EUkuMTPgZcV0QhQrOcwMPcBCcd_Q1&si=8lwPjXPkdAmja2FR

RAG (Retrieval-Augmented Generation) Series

· YouTube playlist: https://youtu.be/lxzWJaL9HB0?si=naJRUL83-ExSyLew
· Covers advanced RAG production issues

Roadmaps

1. Junior MLOps roadmap
2. DevOps to MLOps roadmap

· Links provided in community messages

Important Technical Concepts Discussed

1. Pickle and Model Serialization

· Safety concerns with pickle
· Alternative: ONNX for model serialization
· LinkedIn post covers details

2. ONNX (Open Neural Network Exchange)

· Framework for model serialization
· Allows using models across different frameworks

3. TensorRT

· For accelerating inference performance
· Useful when inference is slow

4. API Performance

· ML Model API optimization
· CI/CD with build and test automation

Community Support

GitHub Repositories

· Mini-Project Example: https://github.com/Mostafa-Ashraf-1/Uplifting-Mini-Project
· Contains: Classes, subroutines, logs, tests

Discord Community

· Link: https://discord.gg/7wTNcR4kz

LinkedIn Community

· MLOps Mena Page: Regular posts with resources, tips, and explanations
· Hashtags: #MLOps #MachineLearning #AI

Feedback & Improvement

· Docker Session Feedback Form: https://docs.google.com/forms/d/e/1FAIpQLSfOjbwW_qObDXOVADaxmKNMfs7-cv8Q0RwWwhaRf2F7eaXHng/viewform

Upcoming Events

· Next Docker session: Saturday, August 29
· Regular sessions: Sundays at 7 PM

Certificate Details

1. Attendance Certificate: Fill attendance form in YouTube description for all 5 sessions
2. Completed Certificate: Complete Final Project (not mandatory but recommended)

Best Practices Shared

· Start by understanding the problem and workflow without AI
· Check data sources
· Formulate data for input and output
· Confirm requirements with manager
· Training and evaluation
· Consider deployment and time constraints

Tools Mentioned

· uv & toml: Package management
· DictVectorizer: Feature extraction
· Pickle: Model serialization (with warnings)
· ONNX: Cross-framework model serialization
· TensorRT: Inference optimization
