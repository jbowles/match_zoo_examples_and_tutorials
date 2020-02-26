# #https://github.com/NTMC-Community/MatchZoo
# #https://matchzoo.readthedocs.io/en/master/

# #To train a Deep Semantic Structured Model, import matchzoo and prepare input data.
# #Deep Semantic Structured Model: https://www.microsoft.com/en-us/research/project/dssm/
import matchzoo as mz

train_pack = mz.datasets.wiki_qa.load_data('train', task='ranking')
valid_pack = mz.datasets.wiki_qa.load_data('dev', task='ranking')

# #Preprocess your input data in three lines of code, keep track parameters to be passed into the model.
preprocessor = mz.preprocessors.DSSMPreprocessor()
train_processed = preprocessor.fit_transform(train_pack)
valid_processed = preprocessor.transform(valid_pack)


# #Make use of MatchZoo customized loss functions and evaluation metrics:
ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=4))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.MeanAveragePrecision()
]

# #Initialize the model, fine-tune the hyper-parameters.
model = mz.models.DSSM()
model.params['input_shapes'] = preprocessor.context['input_shapes']
model.params['task'] = ranking_task
model.guess_and_fill_missing_params()
model.build()
model.compile()

# #Generate pair-wise training data on-the-fly, evaluate model performance using customized callbacks on validation data.
# #WARNING: PairDataGenerator will be deprecated in MatchZoo v2.2. Use `DataGenerator` with callbacks instead.
# #train_generator = mz.PairDataGenerator(train_processed, num_dup=1, num_neg=4, batch_size=64, shuffle=True)
train_generator = mz.DataGenerator(train_processed, num_dup=1, num_neg=4, batch_size=64, shuffle=True)
valid_x, valid_y = valid_processed.unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=valid_x, y=valid_y, batch_size=len(valid_x))
history = model.fit_generator(train_generator, epochs=20, callbacks=[evaluate], workers=5, use_multiprocessing=False)
