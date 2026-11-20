# Failure analysis: MPNet (test queries)

260 of 10000 queries (2.6%) have no relevant question in the top 10. Reasons are assigned by simple rules (first match):

| Category | Queries | Share of misses |
|---|---:|---:|
| Other | 111 | 42.7% |
| Look-alike distractor | 96 | 36.9% |
| Paraphrase (low word overlap) | 42 | 16.2% |
| Length mismatch | 11 | 4.2% |

- *Look-alike distractor*: the top-ranked question shares at least half of its words with the query but is not a labelled duplicate. These are often questions that differ in one key detail, or true duplicates that the dataset did not label.
- *Paraphrase*: the labelled duplicate shares few words with the query.
- *Length mismatch*: query and duplicate differ in length by 2x or more.

## Other

**Query:** What are some good books or online tutorials for learning Python from basics?

- Relevant (rank 18 in MPNet): What are best books for learning python?
- Top 3 retrieved by MPNet:
  1. Which is the best books or online courses for learning python from basic to advanced?
  2. What is a good resource to learn Python?
  3. What are the best resources for learning Python?


**Query:** How can l be more concentrated and stop being absent -minded?

- Relevant (rank 19 in MPNet): How do I reduce being absent minded?
- Relevant (rank 17 in MPNet): How can I stop being absent minded?
- Relevant (rank 11 in MPNet): How can I stop being absent minded all the time?
- Relevant (rank >50 in MPNet): How does a person get over being absent minded?
- Top 3 retrieved by MPNet:
  1. How do I stay focused all the time?
  2. What should we do for more concentration?
  3. How do I increase my concentration?


**Query:** How can I find cheap discount coupon online?

- Relevant (rank 17 in MPNet): Is there any site to find cheap tickets with discount coupon?
- Top 3 retrieved by MPNet:
  1. Where is a good place to find online coupon codes?
  2. How and from where can one get discount coupons for websites from all merchants?
  3. How do I get discount coupons for website?


**Query:** Where can I find a hacker to hack a Facebook account?

- Relevant (rank 37 in MPNet): Can someone hack a Facebook account for me?
- Top 3 retrieved by MPNet:
  1. Where can I hire Facebook account hacker and how much should I pay him/her?
  2. Where can I hire a hacker who can find out the real owner of a fake Facebook account?
  3. How can I hack a Facebook account?


**Query:** How can I buy a house with bad credit?

- Relevant (rank 11 in MPNet): How can having bad credit cost you money when buying a house or a car?
- Top 3 retrieved by MPNet:
  1. How does one w/bad credit get a loan against a house they own to buy another property?
  2. How do you buy a house?
  3. How will I be able to buy a house?


## Look-alike distractor

**Query:** Do bad people exist? Or are they just good people making mistakes?

- Relevant (rank >50 in MPNet): Is everybody good at heart, just misunderstood?
- Top 3 retrieved by MPNet:
  1. Do good people exist, or are they just bad people making good things?
  2. Are there good and bad people?
  3. Are there bad people in the world?


**Query:** What is current account and savings account?

- Relevant (rank 13 in MPNet): How do I know whether my account is savings account or current account?
- Top 3 retrieved by MPNet:
  1. What is the difference between current account savings account?
  2. What is the difference between a savings account and a current account?
  3. What's difference between saving account and current account?


**Query:** Can India be a developed country?

- Relevant (rank 26 in MPNet): When India will be fully develop like other developed countries?
- Relevant (rank 46 in MPNet): When will India look like U.S., Germany, France and other developed countries?
- Relevant (rank 11 in MPNet): When will India be a developed country?
- Relevant (rank 23 in MPNet): How long will India take to be a developed country?
- Relevant (rank 33 in MPNet): How long will India be a developing country?
- Top 3 retrieved by MPNet:
  1. Can India be a developed nation?
  2. Is india a developed country?
  3. India as a developed nation?


**Query:** Where are you from and what is your daily routine?

- Relevant (rank 32 in MPNet): What is daily routine?
- Top 3 retrieved by MPNet:
  1. What is your daily routine?
  2. What is your daily routine?
  3. What are your daily routines?


**Query:** What are the best places to go alone in Pune?

- Relevant (rank 29 in MPNet): Which are the best places to visit in Pune in a single day?
- Top 3 retrieved by MPNet:
  1. What are the best places to explore alone in Pune on the weekends?
  2. What are the best places to visit near Pune for solo?
  3. What are the best places in Pune?


## Paraphrase (low word overlap)

**Query:** Will things be worse with a trump presidency?

- Relevant (rank >50 in MPNet): What is the worse Trump can do as president?
- Top 3 retrieved by MPNet:
  1. Would a Donald Trump presidency really be so bad?
  2. If Donald Trump is president, will he turn out to be worse than Barack Obama?
  3. Will the United States be doomed if Donald Trump becomes President?


**Query:** Why doesn't China like Japan?

- Relevant (rank 31 in MPNet): What is the long-standing grudge within China and Japan? How did it start?
- Top 3 retrieved by MPNet:
  1. Why do China and Japan hate each other?
  2. Why do some Chinese people hate Japan?
  3. Why Chinese hate Japanese?


**Query:** How we can speak English well?

- Relevant (rank >50 in MPNet): Why can't I speak English well, even though I have studied it for 11 years?
- Top 3 retrieved by MPNet:
  1. What is the best way to speak English well?
  2. How can l learn English and speak English well?
  3. How can one learn English well?


**Query:** Where can I learn about investing in stock market from scratch?

- Relevant (rank 42 in MPNet): What is the best way to learn stock trading?
- Relevant (rank 47 in MPNet): What is a good way to learn stock trading?
- Relevant (rank >50 in MPNet): How do I learn how to trade stocks by myself?
- Relevant (rank >50 in MPNet): What is the best way to learn how to trade stocks?
- Relevant (rank >50 in MPNet): How can a beginner (that's never traded stocks) learn how to trade stocks?
- Top 3 retrieved by MPNet:
  1. How do I learn about stock market investing?
  2. Where is the best place to get information about investing in stock market for a beginner?
  3. How do I learn stock market investing?


**Query:** What are the best ways to temporarily ease anxiety?

- Relevant (rank 31 in MPNet): How can I ease pressure caused by anxiety?
- Top 3 retrieved by MPNet:
  1. What are good ways to deal with anxiety?
  2. What is the best way to counteract anxiety as soon as it starts?
  3. What should I do to relieve anxiety?


## Length mismatch

**Query:** I want to invest 20k per month in india for 5 years. What are the best options to get maximum out of my investment?

- Relevant (rank >50 in MPNet): I want to invest 30k per month. What are the best options?
- Top 3 retrieved by MPNet:
  1. What is best investment option in India?
  2. What is best investment options in India?
  3. What are best investment options in India?


**Query:** I want to learn about investment, where should I start?

- Relevant (rank 13 in MPNet): I am 17 and want to learn about investing, but nearly everything on the internet is made up. Where should I start?
- Top 3 retrieved by MPNet:
  1. How can I learn investment?
  2. How can I learn how to invest?
  3. How should I start learning about stocks and investments? How do I start?


**Query:** Speed of Light: How fast can we move?

- Relevant (rank 27 in MPNet): Why is the speed of light the limit for how fast a physical object can move?
- Top 3 retrieved by MPNet:
  1. What’s the speed of light?
  2. What is the speed of light?
  3. Can we ever reach the speed of light?


**Query:** What are the chances for Donald Trump to win?

- Relevant (rank 27 in MPNet): What are the chances of Donald Trump winning the presidential election? If he does win, what are the chances he will be impeached?
- Top 3 retrieved by MPNet:
  1. What are the chances of Donald Trump Winning the 2016 election?
  2. What are the odds of Donald Trump winning the presidency?
  3. How likely is it for Donald Trump to win the 2016 US presidential election?


**Query:** Why did the government replace 1000 to 2000?

- Relevant (rank 11 in MPNet): Why did the government print Rs 2000 notes? Why they didn’t print new 1000 notes?
- Top 3 retrieved by MPNet:
  1. Why did Indian government stopped making 1000 rupee note and started making 2000 instead?
  2. Why did the government decide to scrap 1000 rupee notes and introduce a denomination of 2000 instead?
  3. Why was the 1000 rupee note replaced with a 2000 rupee note?


