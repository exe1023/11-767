Lab 2: Project Workshopping / Peer Feedback
===
The goal of this lab is for you to give and receive peer feedback on project outlines before writing up your proposal. 

- **You can find your team's reviewing assignments in the first sheet [here](https://docs.google.com/spreadsheets/d/1_pw_lYkFutMjuL1_j6RdxNyQlj7LvF_f5eEKr1Qm-w0/edit?usp=sharing).**
- **The second sheet contains links to all teams' outlines to review.**
- **Once you have reviewed an outline, please enter a link to your review (e.g. a Google Doc, public github repo, etc.) replacing your team name in the corresponding other team's row in the third sheet.**


Here's a reminder of what your completed project proposal should include:
- Motivation
- Hypotheses (key ideas)
- How you will test those hypotheses: datasets, ablations, and other experiments or analyses.
- Related work and baselines
- I/O: What are the inputs and output modalities? What existing tools will you use to convert device inputs (that are not a core part of your project) to a format readable by the model, and vice versa?
- Hardware, including any peripherals required, and reasoning for why that hardware is needed for this project. (This is where you will request additional hardware and/or peripherals for your project!)
- Will you need to perform training off-device? If so, do you need cloud compute credits (GCP or AWS), and how much?
- Potential challenges, and how you might adjust the project to adapt to those challenges.
- Potential extensions to the project.
- Potential ethical implications of the project.
- Timeline and milestones. (Tip: align with Thursday in-class labs!)

Group name: How do you turn this on 
---
Group members present in lab today: Yi-Ting Yeh, Tin-Ray Chiang

1: Review 1
----
Name of team being reviewed: Triple Fighting
1. How does your team's background align with that of the proposed project (e.g. if the proposed project is in CV but your team's backgorund is mostly NLP, state that here. Similarly, if your team also has someone who specializes in the area of the proposed project, state that here.)
- We both specialize in NLP but have experience in speech recognition. Therefore, our background aligns with the ASR part of the project.
2. Are the motivation, hypotheses, and plan to test those hypotheses clear? If not, where could the team provide more clarification or explanation?
- The motivation, hypotheses are clear.
- It might be better to provide more details on how to test hypotheses and potential difficulties. For example, what might happen if we choose mobilenet as a baseline. How about other CV models and ASR models?
3. Does the project seem properly scoped for a semester-long project? Why or why not? How might the team be able to adjust the project to be appropriately scoped?
- It might be too ambitious to do both image recognition, speech recognition in one project since they are not solved problems. 
- Also, collecting data could also be a challenging task. You might want to only use publicly available datasets if possible.
- For your project, we think it is better to focus on tackling your hypothesis i and ii. Hypothesis iii and iv could be future work.
4. Are there any potential ethical concerns that arise from the proposed project?
- If you want to collect data from random CMU students, there will be some privacy issues.
5. Any additional comments or concerns? Any related work you know of that might be relevant and/or helpful?
- ASR library: https://github.com/espnet/espnet 
2: Review 2
----
Name of team being reviewed: MasterOfScience
1. How does your team's background align with that of the proposed project (e.g. if the proposed project is in CV but your team's backgorund is mostly NLP, state that here. Similarly, if your team also has someone who specializes in the area of the proposed project, state that here.)
- We are both in NLP but this project is in CV that we are not familiar with.
2. Are the motivation, hypotheses, and plan to test those hypotheses clear? If not, where could the team provide more clarification or explanation? 
- The motivation, hypotheses, and methodology are clear.
- It would be better to provide the details of how to test the models e.g. metrics.
3. Does the project seem properly scoped for a semester-long project? Why or why not? How might the team be able to adjust the project to be appropriately scoped?
- We guess the loading of working on both real-time inference and personalized models might be too high.
- You might want to choose either topic to be the extension of the project.
- Otherwise, if you can easily run the pretrained model on the device, working on both might be feasible for a semester-long project.
4. Are there any potential ethical concerns that arise from the proposed project? 
- No
5. Any additional comments or concerns? Any related work you know of that might be relevant and/or helpful?

3: Review 3
----
Name of team being reviewed: dst
1. How does your team's background align with that of the proposed project (e.g. if the proposed project is in CV but your team's backgorund is mostly NLP, state that here. Similarly, if your team also has someone who specializes in the area of the proposed project, state that here.)
- Although we are both NLP people, we took Multimodal Machine Learning and our team project was also vision-and-language navigation (worked on ALFRED dataset).
2. Are the motivation, hypotheses, and plan to test those hypotheses clear? If not, where could the team provide more clarification or explanation? 
- Motivation, hypotheses, and how to test hypotheses are clear. 
- It will be better to provide more details on the potential challenges of using Matterport Simulator and corresponding solutions. For example, what if it requires additional ram or GPU resources.
3. Does the project seem properly scoped for a semester-long project? Why or why not? How might the team be able to adjust the project to be appropriately scoped?
- This project should be properly scoped for a semester-long project if you only use the simulator to train and test models.
- The physical navigation in a 3D environment might be too challenging since you need to collect additional data.
4. Are there any potential ethical concerns that arise from the proposed project? 
-  No 
5. Any additional comments or concerns? Any related work you know of that might be relevant and/or helpful?
- Another VLN task to consider: https://arxiv.org/abs/1912.01734 
4: Receiving feedback
----
Read the feedback that the other groups gave on your proposal, and discuss as a group how you will integrate that into your proposal. List 3-5 useful pieces of feedback from your reviews that you will integrate into your proposal:
1. It might be interesting to check for the performance impact of distillation for noisy settings. (From Macrosoft)
2. How will the model handle the cocktail party problem, i.e., recognize different voices from different sources at the same time? (From MSC)
3. more clarifying information could be given regarding each of the given items: e.g. a brief explanation of what is in the Librespeech dataset, what metrics the baselines use, etc. (From dst)
4. For ethical concerns, we think other people should know when you start the ASR system because you will record other people’s speech. Thus, there should be an indication for who you are recording. (From MSC)

You may also use this time to get additional feedback from instructors and/or start working on your proposal.


