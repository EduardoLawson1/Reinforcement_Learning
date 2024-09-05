
# Q-Learning vs SARS in TAXI v3 environment

This project was developed for a subject in my Automation engineering course.
is environment, a taxing of each episode.

The Taxi-v3 environment is a classic problem introduced in “Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition” by Tom Dietterich. 
In this environment, a taxi needs to pick up a passenger from one location and drop them off at another location in a 5x5 grid world. The taxi starts at a random position, and the passenger's location and destination are also chosen randomly at the beginning of each episode.

![](https://github.com/EduardoLawson1/Reinforcement_Learning/blob/main/QlearningVSsarsa/q_learning_policy.gif)


The training results of the Q-Learning and SARSA algorithms in the Taxi-v3 environment revealed significant differences in terms of convergence and performance. Both algorithms were trained for 10,000 episodes, with the main indicators used for evaluation being the cumulative reward per episode and the number of epochs per episode.

![Screenshot from 2024-07-23 20-16-58](https://github.com/user-attachments/assets/61a44d3c-e7f4-462f-af69-91bb36de5c4d)


![Screenshot from 2024-07-23 20-16-38](https://github.com/user-attachments/assets/93b36ba0-b7f9-43bd-bd05-41e792bdeac9)

For the Q-Learning algorithm, the cumulative reward per episode graph started at approximately -3000. During the initial learning phase, the cumulative reward rose rapidly, albeit with significant variation. This rapid rise, despite the noise, indicates that the algorithm was intensively exploring and learning which actions resulted in higher rewards. Stability was achieved around episode 1500, where the cumulative reward stabilized around zero. This suggests that from this point, the agent managed to maintain a stable and efficient policy.

Regarding the epochs per episode graph for Q-Learning, the number of epochs started around 1400 in the initial episode. Similar to the cumulative reward, there was an abrupt drop in the number of epochs, although the graph remained quite noisy until stabilizing around episode 1700. From this point, the number of epochs per episode stabilized at zero, indicating that the agent learned to complete the task efficiently and consistently.

On the other hand, the SARSA algorithm had a more challenging start. The cumulative reward per episode graph began around -4000, showing an initial performance inferior to Q-Learning. The cumulative reward rose abruptly, but the noise remained strong until around episode 1800, when the reward finally stabilized around zero. This later stabilization suggests that SARSA had a slower convergence, likely due to its on-policy approach, which tends to be more conservative and follow more exploratory policies.

The epochs per episode graph for SARSA showed that the number of epochs started around 2300 in the initial episode, indicating significantly less efficient initial performance compared to Q-Learning. There was an abrupt drop in the number of epochs, but the noise persisted until episode 1800, when it finally stabilized at zero. Similar to the cumulative reward, the later stabilization reflects SARSA's slower convergence.

Comparing the two algorithms, Q-Learning demonstrated faster convergence and better initial performance in terms of cumulative reward and number of epochs per episode. The off-policy nature of Q-Learning, which allows for more aggressive exploration and rapid identification of the best actions, explains the rapid rise and stabilization observed in the graphs. In contrast, SARSA, being an on-policy algorithm, updates its estimates based on the actions actually taken, resulting in more conservative and slower learning, especially in noisy or complex environments.
