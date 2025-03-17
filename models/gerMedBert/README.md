---
license: apache-2.0
language:
- de
---
# *medBERT.de*: A Comprehensive German BERT Model for the Medical Domain

*medBERT.de* is a German medical natural language processing model based on the BERT architecture, 
specifically trianed-tuned on a large dataset of medical texts, clinical notes, research papers, and healthcare-related documents. 
It is designed to perform various NLP tasks in the medical domain, such as medical information extraction, diagnosis prediction, and more.

## Model Details:

### Architecture
*medBERT.de* is based on the standard BERT architecture, as described in the original BERT paper 
("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al.).   
The model employs a multi-layer bidirectional Transformer encoder, which allows it to capture contextual information 
from both left-to-right and right-to-left directions in the input text. 
*medBERT.de* has 12 layers, 768 hidden units per layer, 8 attention heads in each layer and can process up to 512 tokens in a single input sequence.  


### Training Data:

**medBERT.de** is fine-tuned on a large dataset of medical texts, clinical notes, research papers, 
and healthcare-related documents. This diverse dataset ensures that the model is well-versed in various medical 
subdomains and can handle a wide range of medical NLP tasks.
The following table provides an overview of the data sources used for pretraining **medBERT.de**:

| Source                      | No. Documents | No. Sentences | No. Words      | Size (MB) |
|-----------------------------|--------------|---------------|----------------|-----------|
| DocCheck Flexikon           | 63,840       | 720,404       | 12,299,257     | 92        |
| GGPOnc 1.0                  | 4,369        | 66,256        | 1,194,345      | 10        |
| Webcrawl                    | 11,322       | 635,806       | 9,323,774      | 65        |
| PubMed abstracts            | 12,139       | 108,936       | 1,983,752      | 16        |
| Radiology reports           | 3,657,801    | 60,839,123    | 520,717,615    | 4,195     |
| Spinger Nature              | 257,999      | 14,183,396    | 259,284,884    | 1,986     |
| Electronic health records   | 373,421      | 4,603,461     | 69,639,020     | 440       |
| Doctoral theses             | 7,486        | 4,665,850     | 90,380,880     | 648       |
| Thieme Publishing Group     | 330,994      | 10,445,580    | 186,200,935    | 2,898     |
| Wikipedia                   | 3,639        | 161,714       | 2,799,787      | 22        |
|-----------------------------|--------------|---------------|----------------|-----------|
| Summary                     | 4,723,010    | 96,430,526    | 1,153,824,249  | 10,372    |

All training data was completely anonymized and all patient context was removed. 

### Preprocessing:
The input text is preprocessed using the WordPiece tokenization technique, which breaks the text into subword units 
to better capture rare or out-of-vocabulary words. We kept the case format and die not remove special characters from the text. 
**medBERT.de** comes with it's own tokenizer, specifically optimized for German medical language. 

## Performance Metrics:
We finetuned **medBERT.de** on a variety of downstream tasks and compared it to other, state of the art BERT models in the German medical domain.   
Here are some exemplary results for classification tasks, based on radiology reports.
Please refer to our paper for more detailled results. 


| Model                              | AUROC   | Macro F1 | Micro F1 | Precision | Recall   |
|------------------------------------|---------|----------|----------|-----------|----------|
| **Chest CT**                       |         |          |          |           |          |
| GottBERT                           | 92.48   | 69.06    | 83.98    | 76.55     | 65.92    |
| BioGottBERT                        | 92.71   | 69.42    | 83.41    | 80.67     | 65.52    |
| Multilingual BERT                  | 91.90   | 66.31    | 80.86    | 68.37     | 65.82    |
| German-MedBERT                     | 92.48   | 66.40    | 81.41    | 72.77     | 62.37    |
| *medBERT.de*                       | **96.69** | **81.46**  | **89.39**  | **87.88**  | **78.77** |
| *medBERT.de*<sub>dedup</sub>       | 96.39   | 78.77    | 89.24    | 84.29     | 76.01    |
| **Chest X-Ray**                    |         |          |          |           |          |
| GottBERT                           | 83.18   | 64.86    | 74.18    | 59.67     | 78.87    |
| BioGottBERT                        | 83.48   | 64.18    | 74.87    | 59.04     | 78.90    |
| Multilingual BERT                  | 82.43   | 63.23    | 73.92    | 56.67     | 75.33    |
| German-MedBERT                     | 83.22   | 63.13    | 75.39    | 55.66     | 78.03    |
| *medBERT.de*                       | **84.65** | **67.06**  | **76.20**  | **60.44**  | **83.08** |
| *medBERT.de*<sub>dedup</sub>       | 84.42   | 66.92    | 76.26    | 60.31     | 82.99    |


## Fairness and Bias

There are several potential biases in the training data for MedBERT, which may impact the model's performance and fairness:
### Geographic Bias
As a significant portion of the clinical data comes from a single hospital located in Berlin, Germany, 
the model may be biased towards the medical practices, terminology, and diseases prevalent in that specific region. 
This can result in reduced performance and fairness when applied to other regions or countries with different healthcare systems and patient populations.

### Demographic Bias
The patient population at the Berlin hospital may not be representative of the broader German or global population. 
Differences in age, gender, ethnicity, and socioeconomic status can lead to biases in the model's predictions and understanding 
of certain medical conditions, symptoms, or treatments that are more common in specific demographic groups.

### Specialty Bias
A large part of the training data consists of radiology reports, which could bias the model towards the language and concepts 
used in radiology. This may result in a less accurate understanding of other medical specialties or subdomains that are 
underrepresented in the training data.

## Security and Privacy

Data Privacy: To ensure data privacy during the training and usage of *medBERT.de*, several measures have been taken:

### Anonymization
All clinical data used for training the model has been thoroughly anonymized, with patient names and other personally 
identifiable information (PII) removed to protect patient privacy. 
Although some data sources, such as DocCheck, may contain names of famous physicians or individuals who gave talks recorded on the DocCheck paltform. 
These instances are unrelated to patient data and should not pose a significant privacy risk. However, it is possible to extract these names form the model. 

All training data is stored securely, and will not be publicly accessible. However, we will make some training data for the medical benchmarks available.

### Model Security
MedBERT has been designed with security considerations in mind to minimize risks associated with adversarial attacks and information leakage.   
We tested the model for information leakage, and no evidence of data leakage has been found. 
However, as with any machine learning model, it is impossible to guarantee complete security against potential attacks.


## Limitations

**Generalization**: *medBERT.de* might struggle with medical terms or concepts that are not part of the training dataset, especially new or rare diseases, treatments, and procedures.  
**Language Bias**: *medBERT.de* is primarily trained on German-language data, and its performance may degrade significantly for non-German languages or multilingual contexts.  
**Misinterpretation of Context**: *medBERT.de* may occasionally misinterpret the context of the text, leading to incorrect predictions or extracted information.  
**Inability to Verify Information**: *medBERT.de* is not capable of verifying the accuracy of the information it processes, making it unsuitable for tasks where data validation is critical.  
**Legal and Ethical Considerations**: The model should not be used to make or take part in medical decisions and should be used for research only. 

# Terms of Use
By downloading and using the MedBERT model from the Hugging Face Hub, you agree to abide by the following terms and conditions:

**Purpose and Scope:** The MedBERT model is intended for research and informational purposes only and must not be used as the sole basis for making medical decisions or diagnosing patients. 
The model should be used as a supplementary tool alongside professional medical advice and clinical judgment.

**Proper Usage:** Users agree to use MedBERT in a responsible manner, complying with all applicable laws, regulations, and ethical guidelines. 
The model must not be used for any unlawful, harmful, or malicious purposes. The model must not be used in clinical decicion making and patient treatment. 

**Data Privacy and Security:** Users are responsible for ensuring the privacy and security of any sensitive or confidential data processed using the MedBERT model. 
Personally identifiable information (PII) should be anonymized before being processed by the model, and users must implement appropriate measures to protect data privacy.

**Prohibited Activities:** Users are strictly prohibited from attempting to perform adversarial attacks, information retrieval, or any other actions that may compromise 
the security and integrity of the MedBERT model. Violators may face legal consequences and the retraction of the model's publication.

By downloading and using the MedBERT model, you confirm that you have read, understood, and agree to abide by these terms of use. 

# Legal Disclaimer: 
By using *medBERT.de*, you agree not to engage in any attempts to perform adversarial attacks or information retrieval from the model. 
Such activities are strictly prohibited and constitute a violation of the terms of use. 
Violators may face legal consequences, and any discovered violations may result in the immediate retraction of the model's publication. 
By continuing to use *medBERT.de*, you acknowledge and accept the responsibility to adhere to these terms and conditions.

# Citation

```
@article{medbertde,
    title={MEDBERT.de: A Comprehensive German BERT Model for the Medical Domain},
    author={Keno K. Bressem and Jens-Michalis Papaioannou and Paul Grundmann and Florian Borchert and Lisa C. Adams and Leonhard Liu and Felix Busch and Lina Xu and Jan P. Loyen and Stefan M. Niehues and Moritz Augustin and Lennart Grosser and Marcus R. Makowski and Hugo JWL. Aerts and Alexander Löser},
    journal={arXiv preprint arXiv:2303.08179},
    year={2023},
    url={https://doi.org/10.48550/arXiv.2303.08179},
    note={Keno K. Bressem and Jens-Michalis Papaioannou and Paul Grundmann contributed equally},
    subject={Computation and Language (cs.CL); Artificial Intelligence (cs.AI)},
}
```
- 