# Qualitative examples (test queries)

Reranking changed the first relevant rank for 2948 of 10000 queries: 1064 improved, 1884 got worse.

## Dense retrieval finds it, BM25 misses it

The labelled duplicate is not in BM25's top 10 but is ranked first by dense retrieval.

*505 queries match.*

**Query:** How do speakers make so many sounds at once?

- Relevant (rank 1 in MPNet): How can speakers play more than one sound at once?
- Top 3 retrieved by MPNet:
  1. How can speakers play more than one sound at once? (relevant)
  2. How do speakers produce sound waves?
  3. How does our ear hear sounds produced from different instruments at the same time?

*BM25 first relevant rank: 11; MPNet: 1; word overlap 0.27*

**Query:** What are some Android hacking tips?

- Relevant (rank 1 in MPNet): What are some Android hacks?
- Top 3 retrieved by MPNet:
  1. What are some Android hacks? (relevant)
  2. What are the best Android hacks?
  3. What are some cool Android phone hacks?

*BM25 first relevant rank: 31; MPNet: 1; word overlap 0.57*

**Query:** What does a good answer on Quora look like? What does it mean to "be helpful"?

- Relevant (rank 1 in MPNet): How do you write a good answer on Quora?
- Relevant (rank 2 in MPNet): How can I write good answers to Quora questions?
- Top 3 retrieved by MPNet:
  1. How do you write a good answer on Quora? (relevant)
  2. How can I write good answers to Quora questions? (relevant)
  3. How do you tell a good answer from a bad answer on Quora?

*BM25 first relevant rank: not in top 50; MPNet: 1; word overlap 0.21*

**Query:** How can I start to writing a novel?

- Relevant (rank 3 in MPNet): What is the best way to start writing novels?
- Relevant (rank 1 in MPNet): How do I get started on writing my novel?
- Top 3 retrieved by MPNet:
  1. How do I get started on writing my novel? (relevant)
  2. I want to write a novel, but I don`t know how to start. What should I do first?
  3. What is the best way to start writing novels? (relevant)

*BM25 first relevant rank: 20; MPNet: 1; word overlap 0.26*

**Query:** What costume should I wear for Halloween?

- Relevant (rank 1 in MPNet): What should I dress as for Halloween?
- Top 3 retrieved by MPNet:
  1. What should I dress as for Halloween? (relevant)
  2. What should I wear for a zombie costume?
  3. What are some of the best Halloween costumes ever?

*BM25 first relevant rank: 14; MPNet: 1; word overlap 0.56*

## The cross-encoder fixes the ranking

The duplicate was not ranked first by MPNet but is ranked first after reranking.

*692 queries match.*

**Query:** If I have an idea for a product, how would I go about patenting it?

- Relevant (rank 1 in MPNet+CE@50): I have an idea for an app, how would I go about patenting it?
- Relevant (rank 5 in MPNet+CE@50): What's the best way to go about getting an idea patented?
- Top 3 retrieved by MPNet+CE@50:
  1. I have an idea for an app, how would I go about patenting it? (relevant)
  2. If I get an idea for an invention but don't know how to make it can I file a patent?
  3. I have some really good ideas that I want to patent. With little money, what are some ways to make my ideas become a reality?

*MPNet first relevant rank: 3; MPNet+CE@50: 1; word overlap 0.46*

**Query:** Why does poop stink?

- Relevant (rank 1 in MPNet+CE@50): Why does my poop stinks?
- Top 3 retrieved by MPNet+CE@50:
  1. Why does my poop stinks? (relevant)
  2. Why is it that your clothes don't absorb poop smell after you poop even though the whole bathroom stinks?
  3. Why does poop smell bad?

*MPNet first relevant rank: 2; MPNet+CE@50: 1; word overlap 0.50*

**Query:** Which is the best job site to get updated openings for an experienced IT professional?

- Relevant (rank 1 in MPNet+CE@50): What are some ways to get updated with the job openings for an experienced IT professional?
- Top 3 retrieved by MPNet+CE@50:
  1. What are some ways to get updated with the job openings for an experienced IT professional? (relevant)
  2. Which are the best job portals in India for experienced professionals?
  3. Which is the best website to find an IT job opening for a fresher?

*MPNet first relevant rank: 2; MPNet+CE@50: 1; word overlap 0.55*

**Query:** What’s the purpose of human life?

- Relevant (rank 3 in MPNet+CE@50): What is the purpose of human life in the world?
- Relevant (rank 4 in MPNet+CE@50): What is humanity's purpose in life?
- Relevant (rank >50 in MPNet+CE@50): What is the purpose of human life with more than 7.2 billion souls on this planet?
- Relevant (rank 1 in MPNet+CE@50): What is the purpose of human life or life in general?
- Relevant (rank 32 in MPNet+CE@50): What is the purpose we serve being human on Earth?
- Top 3 retrieved by MPNet+CE@50:
  1. What is the purpose of human life or life in general? (relevant)
  2. What's the purpose of a human life?
  3. What is the purpose of human life in the world? (relevant)

*MPNet first relevant rank: 2; MPNet+CE@50: 1; word overlap 0.44*

**Query:** What will happen when RBI prints more money than their limit?

- Relevant (rank 1 in MPNet+CE@50): What happens if RBI print notes more than requirement?
- Top 3 retrieved by MPNet+CE@50:
  1. What happens if RBI print notes more than requirement? (relevant)
  2. Who order RBI to print more money?
  3. Will RBI print the exact amount of money it demonetised?

*MPNet first relevant rank: 2; MPNet+CE@50: 1; word overlap 0.25*

## The cross-encoder breaks a correct ranking

The duplicate was ranked first by MPNet but falls lower after reranking.

*1335 queries match.*

**Query:** What would happen if Donald Trump died right now?

- Relevant (rank 2 in MPNet+CE@50): What would happen if Trump died right now?
- Relevant (rank 7 in MPNet+CE@50): What would happen if Donald Trump dies during the election?
- Top 3 retrieved by MPNet+CE@50:
  1. What would happen if Donald Trump died right now? What would happen if Hillary Clinton died right now? Would the other party get the election?
  2. What would happen if Trump died right now? (relevant)
  3. As neither Trump nor Pence have been elected by the Electoral College yet, what would happen if Trump dies between now and then?

*MPNet first relevant rank: 1; MPNet+CE@50: 2; word overlap 0.68*

**Query:** Is PC gaming on the decline?

- Relevant (rank 2 in MPNet+CE@50): Is the era of PC gaming coming to a close?
- Top 3 retrieved by MPNet+CE@50:
  1. Is PC gaming better?
  2. Is the era of PC gaming coming to a close? (relevant)
  3. Is PC gaming better than console gaming?

*MPNet first relevant rank: 1; MPNet+CE@50: 2; word overlap 0.33*

**Query:** What are some mind-blowing Iphone gadgets and tools that most people don't know about?

- Relevant (rank 2 in MPNet+CE@50): What are some mind-blowing iphone tools that most people don't know about?
- Top 3 retrieved by MPNet+CE@50:
  1. What are some mind-blowing iPhone tools that exist that most people don't know about?
  2. What are some mind-blowing iphone tools that most people don't know about? (relevant)
  3. What are some mind-blowing iPhone/Android gadgets that most people don't know about?

*MPNet first relevant rank: 1; MPNet+CE@50: 2; word overlap 0.88*

**Query:** How can I earn money from an online job?

- Relevant (rank >50 in MPNet+CE@50): Is there any online job to earn money?
- Relevant (rank 4 in MPNet+CE@50): How can I earn money online or have an online job?
- Relevant (rank 2 in MPNet+CE@50): How can I earn money by doing online jobs?
- Top 3 retrieved by MPNet+CE@50:
  1. How do I earn money from online jobs? I can do with data entry jobs or photo editing jobs
  2. How can I earn money by doing online jobs? (relevant)
  3. How can I earn money by online working?

*MPNet first relevant rank: 1; MPNet+CE@50: 2; word overlap 0.51*

**Query:** Which language should I learn by myself first: Russian or Spanish?

- Relevant (rank 6 in MPNet+CE@50): Should I learn Russian or Spanish?
- Top 3 retrieved by MPNet+CE@50:
  1. Which language should I learn with Spanish: French or Russian? Why?
  2. Should I learn Spanish, French, or Russian as a second language and why?
  3. Which language should I learn first between French, Italian and Spanish?

*MPNet first relevant rank: 1; MPNet+CE@50: 6; word overlap 0.55*

