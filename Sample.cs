using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

using CNTK;

// sample taken from https://github.com/microsoft/CNTK/blob/987b22a8350211cb4c44278951857af1289c3666/Examples/TrainingCSharp/Common/LogisticRegression.cs
// modified to include logging to tensorboard
namespace Sample
{
    public class TensorBoardFileWriter
    {
        // import InitVec from unmanaged dll
        [DllImport("TensorBoardFileWriter.dll")]
        public static extern void InitVec(IntPtr vector, [MarshalAs(UnmanagedType.LPWStr)] string dir, IntPtr func);
        
        [DllImport("TensorBoardFileWriter.dll")]
        public static extern IntPtr OpenWriter([MarshalAs(UnmanagedType.LPWStr)] string dir);
        
        [DllImport("TensorBoardFileWriter.dll")]
        public static extern void CloseWriter(IntPtr writer);
        
        [DllImport("TensorBoardFileWriter.dll")]
        public static extern void WriteValue(IntPtr writer, [MarshalAs(UnmanagedType.LPWStr)] string name, float value, long step);

        [DllImport("TensorBoardFileWriter.dll")]
        public static extern void Flush(IntPtr writer);
        
        private readonly IntPtr writer;
        public TensorBoardFileWriter(string dir) => this.writer = OpenWriter(dir);
        
        ~TensorBoardFileWriter() => CloseWriter(this.writer);
        
        public void WriteValue(string name, float value, long step) => WriteValue(this.writer, name, value, step);

        public void Flush() => Flush(this.writer);
        
        // create ProgressWriterVector with attached native progress writer
        public static ProgressWriterVector CreateVector(string path, Function network)
        {
            var progress = new ProgressWriterVector();
            var getV = typeof(ProgressWriterVector).GetMethods(BindingFlags.Static|BindingFlags.NonPublic).First(m => m.Name == "getCPtr");
            HandleRef vector = (HandleRef)(getV.Invoke(null, new object[] { progress }));
            
            var getF = typeof(Function).GetMethods(BindingFlags.Static|BindingFlags.NonPublic).First(m => m.Name == "getCPtr");
            HandleRef function = (HandleRef)(getF.Invoke(null, new object[] { network }));
            
            TensorBoardFileWriter.InitVec(vector.Handle, path, function.Handle);
            return progress;
        }
    }

    /// <summary>
    /// This class shows how the train and evaluate a Logistic Regression classifier.
    /// Data are randomly generated into 2 classes with statistically separable features.
    /// See https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_101_LogisticRegression.ipynb for more details.
    /// </summary>
    public class LogisticRegression
    {
        static int inputDim = 3;
        static int numOutputClasses = 2;

        public static void Main()
        {
            Console.WriteLine("Starting training sample");
            TrainAndEvaluate(DeviceDescriptor.GPUDevice(0));
        }

        static public void TrainAndEvaluate(DeviceDescriptor device)
        {
            // build a logistic regression model
            Variable featureVariable = Variable.InputVariable(new int[] { inputDim }, DataType.Float);
            Variable labelVariable = Variable.InputVariable(new int[] { numOutputClasses }, DataType.Float);
            var classifierOutput = CreateLinearModel(featureVariable, numOutputClasses, device);
            var loss = CNTKLib.CrossEntropyWithSoftmax(classifierOutput, labelVariable);
            var evalError = CNTKLib.ClassificationError(classifierOutput, labelVariable);

            // prepare for training
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(0.02, 1);
            IList<Learner> parameterLearners =
                new List<Learner>() { Learner.SGDLearner(classifierOutput.Parameters(), learningRatePerSample) };

            var progressWriter = new TensorBoardFileWriter("log/test");
            var progressWriterVector = TensorBoardFileWriter.CreateVector("log/main", classifierOutput);
            var trainer = Trainer.CreateTrainer(classifierOutput, loss, evalError, parameterLearners, progressWriterVector);

            int minibatchSize = 64;
            int numMinibatchesToTrain = 1000;
            int updatePerMinibatches = 50;

            // train the model
            Random random = new Random(0);
            for (int minibatchCount = 0; minibatchCount < numMinibatchesToTrain; minibatchCount++)
            {
                Value features, labels;
                GenerateValueData(minibatchSize, inputDim, numOutputClasses, out features, out labels, device);
                //TODO: sweepEnd should be set properly instead of false.
#pragma warning disable 618
                trainer.TrainMinibatch(
                    new Dictionary<Variable, Value>() { { featureVariable, features }, { labelVariable, labels } }, device);
#pragma warning restore 618
                PrintTrainingProgress(trainer, minibatchCount, updatePerMinibatches);

                progressWriter.WriteValue("random1", (float)random.Next(), minibatchCount);
                progressWriter.WriteValue("random2", (float)random.Next(), minibatchCount);
                progressWriter.Flush();
            }

            // test and validate the model
            int testSize = 100;
            Value testFeatureValue, expectedLabelValue;
            GenerateValueData(testSize, inputDim, numOutputClasses, out testFeatureValue, out expectedLabelValue, device);

            // GetDenseData just needs the variable's shape
            IList<IList<float>> expectedOneHot = expectedLabelValue.GetDenseData<float>(labelVariable);
            IList<int> expectedLabels = expectedOneHot.Select(l => l.IndexOf(1.0F)).ToList();

            var inputDataMap = new Dictionary<Variable, Value>() { { featureVariable, testFeatureValue } };
            var outputDataMap = new Dictionary<Variable, Value>() { { classifierOutput.Output, null } };
            classifierOutput.Evaluate(inputDataMap, outputDataMap, device);
            var outputValue = outputDataMap[classifierOutput.Output];
            IList<IList<float>> actualLabelSoftMax = outputValue.GetDenseData<float>(classifierOutput.Output);
            var actualLabels = actualLabelSoftMax.Select((IList<float> l) => l.IndexOf(l.Max())).ToList();
            int misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();

            Console.WriteLine($"Validating Model: Total Samples = {testSize}, Misclassify Count = {misMatches}");
        }

        private static void PrintTrainingProgress(Trainer trainer, int minibatchIdx, int outputFrequencyInMinibatches)
        {
            if ((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
            {
                float trainLossValue = (float)trainer.PreviousMinibatchLossAverage();
                float evaluationValue = (float)trainer.PreviousMinibatchEvaluationAverage();
                Console.WriteLine($"Minibatch: {minibatchIdx} CrossEntropyLoss = {trainLossValue}, EvaluationCriterion = {evaluationValue}");
            }
        }

        private static void GenerateValueData(int sampleSize, int inputDim, int numOutputClasses,
            out Value featureValue, out Value labelValue, DeviceDescriptor device)
        {
            float[] features;
            float[] oneHotLabels;
            GenerateRawDataSamples(sampleSize, inputDim, numOutputClasses, out features, out oneHotLabels);

            featureValue = Value.CreateBatch<float>(new int[] { inputDim }, features, device);
            labelValue = Value.CreateBatch<float>(new int[] { numOutputClasses }, oneHotLabels, device);
        }

        private static void GenerateRawDataSamples(int sampleSize, int inputDim, int numOutputClasses,
            out float[] features, out float[] oneHotLabels)
        {
            Random random = new Random(0);

            features = new float[sampleSize * inputDim];
            oneHotLabels = new float[sampleSize * numOutputClasses];

            for (int sample = 0; sample < sampleSize; sample++)
            {
                int label = random.Next(numOutputClasses);
                for (int i = 0; i < numOutputClasses; i++)
                {
                    oneHotLabels[sample * numOutputClasses + i] = label == i ? 1 : 0;
                }

                for (int i = 0; i < inputDim; i++)
                {
                    features[sample * inputDim + i] = (float)GenerateGaussianNoise(3, 1, random) * (label + 1);
                }
            }
        }

        /// <summary>
        /// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        /// https://stackoverflow.com/questions/218060/random-gaussian-variables
        /// </summary>
        /// <returns></returns>
        static double GenerateGaussianNoise(double mean, double stdDev, Random random)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double stdNormalRandomValue = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + stdDev * stdNormalRandomValue;
        }

        private static Function CreateLinearModel(Variable input, int outputDim, DeviceDescriptor device)
        {
            int inputDim = input.Shape[0];
            var weightParam = new Parameter(new int[] { outputDim, inputDim }, DataType.Float, 1, device, "w");
            var biasParam = new Parameter(new int[] { outputDim }, DataType.Float, 0, device, "b");

            return CNTKLib.Times(weightParam, input) + biasParam;
        }
    }
}