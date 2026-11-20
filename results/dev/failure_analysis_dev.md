# Failure analysis: MPNet (val queries)

28 of 1000 queries (2.8%) have no relevant question in the top 10. Reasons are assigned by simple rules (first match):

| Category | Queries | Share of misses |
|---|---:|---:|
| Look-alike distractor | 13 | 46.4% |
| Other | 8 | 28.6% |
| Paraphrase (low word overlap) | 6 | 21.4% |
| Length mismatch | 1 | 3.6% |

- *Look-alike distractor*: the top-ranked question shares at least half of its words with the query but is not a labelled duplicate. These are often questions that differ in one key detail, or true duplicates that the dataset did not label.
- *Paraphrase*: the labelled duplicate shares few words with the query.
- *Length mismatch*: query and duplicate differ in length by 2x or more.

## Look-alike distractor

**Query:** What is the coolest programming language to learn?

- Relevant (rank >50 in MPNet): What is the best programming language to learn first?
- Top 3 retrieved by MPNet:
  1. What is the most fun programming language to learn?
  2. What is the most interesting programming language?
  3. What is the best programming language to learn?


**Query:** Scary Situations: What is the scariest thing you've ever seen?

- Relevant (rank 11 in MPNet): What is the scariest thing you have ever come across?
- Top 3 retrieved by MPNet:
  1. What is the scariest thing you've ever seen?
  2. What's the scariest thing you've ever heard or witnessed?
  3. What is the scariest situation you have ever been in?


**Query:** What exactly is success?

- Relevant (rank >50 in MPNet): Who exactly is successful?
- Top 3 retrieved by MPNet:
  1. What is success?
  2. What is success?
  3. What is the real meaning of Success?


**Query:** Can someone help me solve this high-school math problem?

- Relevant (rank 33 in MPNet): I'm currently stuck with this high-school problem. Can someone help me?
- Top 3 retrieved by MPNet:
  1. can someone please solve this math problem for me?
  2. Can somebody help me with this math problem?
  3. Can anyone help with this math problem?


**Query:** What should I eat immediately after workout?

- Relevant (rank 20 in MPNet): What should I do immediately after workout?
- Top 3 retrieved by MPNet:
  1. What should I eat before and after a workout?
  2. Should I eat after a workout?
  3. How long should I wait until I eat something after a workout and what should I eat?


## Other

**Query:** What can I do for society?

- Relevant (rank 13 in MPNet): What can do for this society?
- Top 3 retrieved by MPNet:
  1. How can I contribute towards society?
  2. How do I contribute to society?
  3. How can I contribute to the society?


**Query:** How does law of attraction work?

- Relevant (rank 28 in MPNet): Does Law of Attraction really work? If yes what have you gained out of it.
- Relevant (rank 11 in MPNet): Does law of attraction really works?
- Top 3 retrieved by MPNet:
  1. What is the law of attraction?
  2. How does law of attraction work? Is it real?
  3. What is the main secret behind the law of attraction?


**Query:** Why should one get married?

- Relevant (rank 16 in MPNet): Should one get married?
- Top 3 retrieved by MPNet:
  1. Why should we get married in our life? Is there any important reason?
  2. Why should someone Marry?
  3. Why do people get married?


**Query:** What is the best way to change mood? I don't feel like doing anything these days :/

- Relevant (rank 13 in MPNet): How can I get out of a, "I don't feel like doing anything" mood?
- Top 3 retrieved by MPNet:
  1. What is the best way to get out of a bad mood?
  2. What do you do to get over a bad mood?
  3. How can I snap myself out of a bad mood?


**Query:** What are some tips for publishing a book I’ve written?

- Relevant (rank 25 in MPNet): What is the procedure of publishing the book self written?
- Top 3 retrieved by MPNet:
  1. What steps do I need to take to write, publish and market a book I've written?
  2. I am writing a book and want to get it published. How do I go about it?
  3. I am a writer and I want to publish a book, how do I go about it?


## Paraphrase (low word overlap)

**Query:** What does the Clinton Foundation do?

- Relevant (rank 12 in MPNet): Will the Clinton Foundation be a problem for Hillary Clinton running for president?
- Top 3 retrieved by MPNet:
  1. How is the Clinton Foundation funded?
  2. What has the Clinton Foundation achieved with the $2 billion of funds raised?
  3. How much Clinton Foundation money goes to charity?


**Query:** How can I make enormous wealth through internet?

- Relevant (rank >50 in MPNet): What are some ways to passively make income through the internet?
- Top 3 retrieved by MPNet:
  1. How do I get rich online?
  2. How can I become millionaire online?
  3. How can I start making money using internet?


**Query:** Is increasing height after the age of 20 possible?

- Relevant (rank 28 in MPNet): Can I grow height after 19 years?
- Relevant (rank 18 in MPNet): Is it possible to increase your height after 19?
- Relevant (rank 32 in MPNet): Is it possible to increase our height after we cross our teenage?
- Relevant (rank 37 in MPNet): Can I increase my body height after 19 ? If yes? Then how?
- Relevant (rank >50 in MPNet): Is it possible to increase height naturally after 19 for females? What exercises we should focus on?
- Top 3 retrieved by MPNet:
  1. Is it possible to increase height after 20?
  2. Is it possible to Increase height or get taller after 20?
  3. Can a person increase his/her height even after 20?


**Query:** How much happy are you with Narendra Modi government?

- Relevant (rank 20 in MPNet): How good is Narendra Modi at governance?
- Top 3 retrieved by MPNet:
  1. Politics: Who is satisfied with Indian Prime Minister Mr. Modi?
  2. How is Modi government doing?
  3. How is Modi's government doing?


**Query:** Is artificial consciousness possible?

- Relevant (rank 23 in MPNet): Is self aware Artificial Intelligence possible?
- Relevant (rank 35 in MPNet): Can artificial intelligence in a simulated reality become self-aware?
- Relevant (rank >50 in MPNet): How can self-awareness theoretically affect an artificial intelligence in the real-world?
- Relevant (rank >50 in MPNet): Could a self-aware artificial intelligence theoretically be "grown"?
- Relevant (rank >50 in MPNet): What is the probability that self-aware artificial intelligence might emerge unintentionally?
- Top 3 retrieved by MPNet:
  1. Is it possible to create machines/robots that are conscious?
  2. Would a human level AI be conscious?
  3. Is consciousness possible without self-awareness?


## Length mismatch

**Query:** Which is the best programming language to learn for hacking? What are some books for beginners?

- Relevant (rank >50 in MPNet): Which programming language should a beginner learn first?
- Relevant (rank >50 in MPNet): What's the best programming language for beginners?
- Top 3 retrieved by MPNet:
  1. Which programming language I should learn for learning hacking?
  2. What is the best programming language I must learn before starting to learn hacking?
  3. What's the best language to learn for hacking, and what do you recommend?


