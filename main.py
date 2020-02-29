from dataset_transformer import Dataset_transformer, nextBatchData
import transformer as trfm
from config import DataConfig as dcfg, ModelConfig as mcfg, TrainingConfig as tcfg
import tensorflow.compat.v1 as tf
import os

dataset = Dataset_transformer()
dataset.dataGen()

trainReviews = dataset.trainReviews
trainLables = dataset.trainLables

validReviews = dataset.validReviews
validLables = dataset.validLables

wordEmbedding = dataset.wordEmbedding

embeddedPosition = trfm.genFixedPositionEmbedding(dcfg.batchSize, dcfg.sequenceLength)

with tf.Graph().as_default():
    tfConfig = tf.ConfigProto()
    tfConfig.gpu_options.allow_growth = True
    session = tf.Session(config=tfConfig)

    with session.as_default():
        transformer = trfm.Transformer(wordEmbedding)

        gloablStep = tf.Variable(0, name='gloableStep', trainable=False)

        optimizer = tf.train.AdamOptimizer(tcfg.learningRate)

        gradsAndVars = optimizer.compute_gradients(transformer.loss)

        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=gloablStep)

        gradSummaries = []

        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram('{}/grad/list'.format(v.name), g)
                tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))

        outDir = os.path.abspath(os.path.join(os.path.curdir, 'summarys'))

        lossSummary = tf.summary.scalar('loss', transformer.loss)

        summaryOp = tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir, 'train')
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, session.graph)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        savedModelPath = "./model/transformer/savedModel"
        if os.path.exists(savedModelPath):
            os.rmdir(savedModelPath)
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)
        session.run(tf.global_variables_initializer())

        def trainStep(batchX, batchY):
            feed_dict = {
                transformer.inputx: batchX,
                transformer.inputy:batchY,
                transformer.dropoutKeepProb: mcfg.dropoutKeepProb,
                transformer.embeddedPosition: embeddedPosition
            }

            _, summary, step, loss, predictions = session.run(
                [trainOp, summaryOp, gloablStep, transformer.loss, transformer.predictions], feed_dict
            )

            trainSummaryWriter.add_summary(summary, step)

            return loss

        for i in range(tcfg.epoches):
            for batchTrain in nextBatchData(trainReviews, trainLables, dcfg.batchSize):
                loss = trainStep(batchTrain[0], batchTrain[1])
                currentStop = tf.train.global_step(session, gloablStep)

        inputs = {"inputX": tf.saved_model.utils.build_tensor_info(transformer.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(transformer.dropoutKeepProb)}

        outputs = {"predictions": tf.saved_model.utils.build_tensor_info(transformer.predictions)}

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={"predict": prediction_signature},
                                             legacy_init_op=legacy_init_op)

        builder.save()


