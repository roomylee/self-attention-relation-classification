# Self Attention for Relation Classification

![model](https://user-images.githubusercontent.com/15166794/53145669-73047b80-35e4-11e9-9778-43447530371d.png)

This repository contains the official TensorFlow implementation of the following paper:

> **Improving the Performance and Interpretability of Semantic Relation Classification with Self-Attention Mechanism**<br>
> Joohong Lee, Sangwoo Seo, Yong Suk Choi<br>
> [paper](https://www.dbpia.co.kr/Journal/ArticleDetail/NODE07613662)


## Usage
### Train / Test
* Train data is located in "*<U>SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT*</U>".
* You can apply some pre-trained word embeddings: [word2vec](https://code.google.com/archive/p/word2vec/), [glove100](https://nlp.stanford.edu/projects/glove/), [glove300](https://nlp.stanford.edu/projects/glove/), and [elmo](https://tfhub.dev/google/elmo/1). The pre-trained files should be located in `resource/`. [Check this code](https://github.com/roomylee/entity-aware-relation-classification/blob/f77668088210ce2bb0e94033bdf1cabb45c0bbf0/train.py#L115).
* In every evaluation step, the test performance is evaluated by test dataset located in "*<U>SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT*</U>".

##### Display help message:
```bash
$ python train.py --help
```
##### Train Example:
```bash
$ python train.py --embeddings glove300
```


## SemEval-2010 Task #8
* Given: a pair of *nominals*
* Goal: recognize the semantic relation between these nominals.
* Example:
	* "There were apples, **<U>pears</U>** and oranges in the **<U>bowl</U>**." 
		<br> → *CONTENT-CONTAINER(pears, bowl)*
	* “The cup contained **<U>tea</U>** from dried **<U>ginseng</U>**.” 
		<br> → *ENTITY-ORIGIN(tea, ginseng)*


### The Inventory of Semantic Relations
1. *Cause-Effect(CE)*: An event or object leads to an effect(those cancers were caused by radiation exposures)
2. *Instrument-Agency(IA)*: An agent uses an instrument(phone operator)
3. *Product-Producer(PP)*: A producer causes a product to exist (a factory manufactures suits)
4. *Content-Container(CC)*: An object is physically stored in a delineated area of space (a bottle full of honey was weighed) Hendrickx, Kim, Kozareva, Nakov, O S´ eaghdha, Pad ´ o,´ Pennacchiotti, Romano, Szpakowicz Task Overview Data Creation Competition Results and Discussion The Inventory of Semantic Relations (III)
5. *Entity-Origin(EO)*: An entity is coming or is derived from an origin, e.g., position or material (letters from foreign countries)
6. *Entity-Destination(ED)*: An entity is moving towards a destination (the boy went to bed) 
7. *Component-Whole(CW)*: An object is a component of a larger whole (my apartment has a large kitchen)
8. *Member-Collection(MC)*: A member forms a nonfunctional part of a collection (there are many trees in the forest)
9. *Message-Topic(CT)*: An act of communication, written or spoken, is about a topic (the lecture was about semantics)
10. *OTHER*: If none of the above nine relations appears to be suitable.


### Distribution for Dataset
* **SemEval-2010 Task #8 Dataset [[Download](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?layout=list&ddrp=1&sort=name&num=50#)]**

	| Relation           | Train Data          | Test Data           | Total Data           |
	|--------------------|:-------------------:|:-------------------:|:--------------------:|
	| Cause-Effect       | 1,003 (12.54%)      | 328 (12.07%)        | 1331 (12.42%)        |
	| Instrument-Agency  | 504 (6.30%)         | 156 (5.74%)         | 660 (6.16%)          |
	| Product-Producer   | 717 (8.96%)         | 231 (8.50%)         | 948 (8.85%)          |
	| Content-Container  | 540 (6.75%)         | 192 (7.07%)         | 732 (6.83%)          |
	| Entity-Origin      | 716 (8.95%)         | 258 (9.50%)         | 974 (9.09%)          |
	| Entity-Destination | 845 (10.56%)        | 292 (10.75%)        | 1137 (10.61%)        |
	| Component-Whole    | 941 (11.76%)        | 312 (11.48%)        | 1253 (11.69%)        |
	| Member-Collection  | 690 (8.63%)         | 233 (8.58%)         | 923 (8.61%)          |
	| Message-Topic      | 634 (7.92%)         | 261 (9.61%)         | 895 (8.35%)          |
	| Other              | 1,410 (17.63%)      | 454 (16.71%)        | 1864 (17.39%)        |
	| **Total**          | **8,000 (100.00%)** | **2,717 (100.00%)** | **10,717 (100.00%)** |


