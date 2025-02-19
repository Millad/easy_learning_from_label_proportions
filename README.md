
# Easy Learning From Label Proportions
## A GDPR friendly method of training a simple text Classifier Without Knowing the Labels of Individual sentences — Learning from label proportions (LLP) in Action (Paper explained by Millad Dagdoni for developers)

![Bags with proportions of different colourful balls](https://github.com/Millad/easy_learning_from_label_proportions/blob/main/bags_image.jpeg?raw=true "Bags with proportions of different colourful balls")

As a software developer, I prioritize user privacy. While general statistical data is usually allowed and necessary for business purposes, collecting detailed user information is not.

Is it possible to extract valuable insights from limited statistical data? Absolutely. My research led me to "Learning from Label Proportions," a machine learning technique applicable to various domains — even text analysis — and especially useful when data access is restricted.

In traditional supervised machine learning, a model is trained using labeled data, where inputs (X) are paired with their corresponding outputs (Y). The goal is to predict Y given a new X.

In contrast, Learning from Label Proportions (LLP) only provides access to "bags" of unlabeled feature vectors, each associated with a proportion representing the positive output for that bag. Detailed individual labels are unavailable, limiting access to granular training data.

These types of solutions have been widely applied in various fields, including commerce, fraud detection, medical databases, and even high-energy physics. While the underlying method dates back to 2005, this article focuses on a more recent adaptation that leverages neural networks for enhanced performance. The core motivation remains the same: enabling learning in scenarios where individual data access is restricted or prohibited.

#### Traditional learning versus LLP (Learning from Label Proportions) 
In traditional machine learning, model training involves comparing predicted outputs (predicted_y) with actual values (goal_y). This comparison helps assess the model's accuracy during training. This approach requires knowing the true y value for each set of x features, necessitating extensive data collection for accurate predictions. Companies like Meta (Facebook) and Google utilize this method to extensively profile and understand their users. 

The text data might look like this with a sentiment level of 1 for positive and 0 for negative:  
**1:** I enjoyed this movie a lot. I have watched it many times. **positive_sentiment:** 1.   
**2:** It’s not that entertaining. I hated every bit of it. I don’t recommend it. **positive_sentiment:** 0.   
**3:** One of the best movies ever. You have to watch it right away. **positive_sentiment:** 1.   

In LLP, you have groups of input features (x) organized into "bags," and you know the proportion of positive outcomes (true Y) for each bag as a probability. The goal is to train a model that identifies patterns across these bags and can predict the correct output (y value) for individual input features, not just for new bags. Example of training data to compare with the previews example above:   
**bag_1:** I enjoyed this movie a lot. I have watched it many times.   
**bag_1:** It’s not that entertaining. I hated every bit of it. I don’t recommend it.  
**bag_1:** One of the best movies ever. You have to watch it right away.   
**Bag_1_label_proportion**: **(2/3)** -> here 2 are positive of 3 total in current bag.  

#### How does the learning part in Learning from label proportions work?
Consider a teacher forming and evaluating numerous student groups. By rearranging students and observing group performance, the teacher can identify patterns and predict high-performing students.

This analogy illustrates the core concept of LLP. The goal is to create a model that learns to predict the average performance (mean) of each group and the proportion of positive outcomes within that group.

The provided loss function aligns with this concept. We aim to predict a mean that ultimately matches the overall average of all groups. While the technical proof is detailed in the paper, the focus here is on the simplicity and practicality of this approach.

#### The Code
https://github.com/Millad/easy_learning_from_label_proportions
#### The Paper 
https://arxiv.org/abs/2302.03115

I will not jump into deep details about proofs here because the proofs are very well explained on the last pages of the paper linked here. The core concept is straightforward, despite the math. Simple expected values and algebra will lead you to **Definition 3.4** shown below. For further exploration of the function mentioned, please checkout and debug the code notebook linked here.

#### Symbols meaning from the paper:
**K**= bag size.  
**X**= features (age, sex, ticket class, level or word vectors…etc).  
**Y**= label space (0,1 = dead or alive, negative or positive).  
**B** = bag og sample k size = {x1, x2, x3, xk}.  
**a** = associated label proportion.  
**S** = collection of features, and proportions.  
**p** = probability of drawing a sample with label y = 1.  
**g** = a arbitrary function / estimator X x Y => R^d. 
**g_hat** = soft label corrected function. 
**y_hat** = predicted y. 
**Pr(I = i)** = *1/k* = represents the probability of selecting the i-th data point as the source for the correct label.  
**h** = hypothesis function. 
**L(h)**= loss function.   
**j** = j is an element of K which it uses as the current variable in a loop of K length.   
***E*** = expected value (*mean*).   

**Definition 3.4**.    
![Image of the soft label proportions loss function definition 3.4 from the paper](https://github.com/Millad/easy_learning_from_label_proportions/blob/main/equation_used_in_ellp_loss.png?raw=true "Image of the soft label proportions loss function definition 3.4 from the paper")

This method reduces the variance of the estimator using soft labels (bag-level proportions). It focuses on learning key patterns from groups of data (bags), similar to the teacher-student analogy. Instead of individual labels, you have proportions (e.g., 70% of customers made a purchase). This ensures that the expected value (mean) of the transformed function remains unbiased. However, it's crucial to be mindful of potential bias in the proportions and remember that inaccurate input data will lead to inaccurate output. 
The code has a running example that uses this. Please take a look.

#### GDPR friendly
Why use label proportions instead of actual labels, especially under GDPR? 
Even if users give consent, GDPR still requires minimization of personal data collection.

Storing actual labels per user can be considered personal data if labels are linked to individuals (e.g., “User X is depressed”). But proportions are aggregated and not tied to individuals, making them easier to handle under GDPR.

Actual label: “This specific user is likely depressed who lives at this address.” (Potentially personal data)

LLP approach: “30% of users in this large region exhibit signs of depression.” (Aggregate, anonymous)

LLP avoids direct personal profiling, reducing legal risks and regulatory complexity.
Getting individual labels can be expensive and prone to bias.

Users might give incorrect answers due to social desirability bias. In some cases, users may refuse to participate if asked for sensitive labels (e.g., mental health, political views). Collecting only proportions simplifies the process and ensures a broader dataset.

If you ask users, “Do you have anxiety?”, some might not answer truthfully. Instead, collecting “How many in this group report anxiety?” is more reliable. LLP reduces self-reporting bias and improves data quality.

LLP allows data collection at scale without storing identifiable information. Proportions can be gathered via surveys, estimates, or weak labeling, avoiding personal tracking.
Easier to train models across different countries without violating local data protection laws (e.g., GDPR in EU, CCPA in California).    

Instead of tracking which customers like a product, LLP can estimate: “In France, 65% of customers prefer Product A over Product B.” Makes global compliance easier and avoids needing country-specific data handling policies.

Some users just won't give consent even if you ask for consent, many users will refuse due to privacy concerns. Proportion-based learning allows you to still collect useful aggregate data without relying on explicit consent.

More users participate when they know their individual responses aren’t stored.
Avoids low participation rates due to privacy fears.

*Thanks for reading*   
Millad Dagdoni
