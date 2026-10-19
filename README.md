# Recommendation Algorithms 

**Implementation:**

* Matrix Factorization Algorithms entirely by hand:  
  * Stochastic Gradient Descent 

**Datasets:**

* Last.fm Music Recommendation Dataset, 17,559,530 tuples of \<user, artist, plays\> \[1\]  
* **Develop on 20,000 users**

**Evaluation:**

* Recommending artists (targets) based on user listening data (feature)  
* Recording accuracy based on whether the user implicitly likes the predicted artist in held out data  
* Comparing accuracy and speed to implicit github library \[3\]

**Tests and Quality:**

* Clear frozen split of train, validation, and test data  
* Predicting artists not featured within the user profile presented to algorithm (testing)

**Team Plan:**

* A: Data cleaning and tidying \+ main algorithm implementation  
* B: Data splits \+ main algorithm implementation  
* C: Tests, benchmarks, comparisons, oversight / help where needed

**Milestones and Deliverables:**

* W0: Proposal feedback and fixes  
* W1: Data cleaning and tidying. Data splits. Theoretical comprehension of algorithms.  
* W2: Solid progress on implementations. Help each other where necessary.  
* W3: Implementation frameworks complete.  
* W4: Benchmarks and report.

**Reference Implementations/Data:**

1. [http\://ocelma.net/MusicRecommendationDataset/lastfm-360K.html](http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html)  
2. [https\://github.com/benfred/implicit](https://github.com/benfred/implicit?tab=readme-ov-file)  
3. [https\://developers.google.com/machine-learning/recommendation/collaborative/matrix](https://developers.google.com/machine-learning/recommendation/collaborative/matrix) 
