# AntibodyFvTm50Predictor
MRes project - prediction the melting temperature of antibodies using machine learning techniques

In this work, AntiBERTy was used to encode antibody sequences. A 2D plot generated using t-SNE shows distinct separation between species based on the 512-feature encoding and a reduced, 60-feature encoding. Using a Gradient Boosting Tree (GBT) method on a blind test dataset, the model showed a Pearson Correlation Coefficient of 0.83 and mean absolute error of 2.69°C between predicted and actual Tm50.

The results from this study demonstrate that, firstly, AntiBERTy can capture unique, species specific features from amino acid sequences, and secondly, an accurate Tm50 predictor can be achieved using a GBT model trained on AntiBERTy encoded amino acid sequences.

A regression model using GBT was shown to be effective in predicting Tm50 with a MAE of 2.61C, PCC 0.84.
![GBT_act_vs_pred_2](https://github.com/Mike-Skehan/AntibodyFvTm50Predictor/assets/97400544/ad04df54-9207-4afe-b96d-d4b66db1b7c2)

AntiBERTy can capture unique, species specific features from amino acid sequences, even after feature reduction form 512-features to 60.

512-features

![TSNE2D_AbYsis_full_berty_per50](https://github.com/Mike-Skehan/AntibodyFvTm50Predictor/assets/97400544/eb684ebc-60c1-4555-ac16-4d709db4ba8c)

60-features

![TSNE2D_AbYsis_60_berty_per50](https://github.com/Mike-Skehan/AntibodyFvTm50Predictor/assets/97400544/f893d021-7e7d-4c3c-9864-ed3510234e20)

Currently, the webapp is not hosted online but can be trialled using Flask to run on a local server using flask --app main run when in the webapp directory.
![webapp](https://github.com/Mike-Skehan/AntibodyFvTm50Predictor/assets/97400544/7e537ac7-b3ab-4177-ba74-e12e6c0fd09a)
