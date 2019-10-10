using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace NeuralNet

{
/// <summary>
/// Place holder for an image sample.
/// </summary>
    public class Image
    {
        // getter and setter for the image label.
        public int Label { get; set; }

        // getter and setter for the image pixel
        public byte[,] Data { get; set; }
    }


    /// <summary>
    /// MnistReader read both training and test data. Read training, test, traning label, test label byte file and
    /// create a collection of image object which consist of an array of pixel values and a label integer.
    /// </summary>
    public static class MnistReader
    {
       
        public const string TestImagesPath = @"\\Mac\Home\Desktop\mnist\mnist\t10k-images-idx3-ubyte";
        public const string TestLabelsPath = @"\\Mac\Home\Desktop\mnist\mnist\t10k-labels-idx1-ubyte";
        public const string TrainImagesPath = @"\\Mac\Home\Desktop\mnist\mnist\train-images-idx3-ubyte";
        public const string TrainLabelsPath = @"\\Mac\Home\Desktop\mnist\mnist\train-labels-idx1-ubyte";

      
        /// <summary>
        /// 
        /// </summary>
        /// <param name="imagesPath"></param>
        /// <param name="labelsPath"></param>
        /// <returns></returns>
        public static IEnumerable<Image> ReadData(string imagesPath, string labelsPath)
        {
            BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            int magicNumber = images.ReadBigInt32();
            int numberOfImages = images.ReadBigInt32();
            int width = images.ReadBigInt32();
            int height = images.ReadBigInt32();

            int magicLabel = labels.ReadBigInt32();
            int numberOfLabels = labels.ReadBigInt32();

            for (int i = 0; i < numberOfImages; i++)
            {
                var bytes = images.ReadBytes(width * height);
                var arr = new byte[height, width];

                arr.ForEach((j, k) => arr[j, k] = bytes[j * height + k]);

                yield return new Image()
                {
                    Data = arr,
                    Label = Convert.ToInt32(labels.ReadByte())

                };
            }
            labels.Close();
            images.Close();
        }
    }
    /// <summary>
    /// The Neuron class represents an artificial neuron
    /// </summary>
    public class Neuron
    {
        public int numberInputs;
        public double bias;
        public double output;
        public double errorGradient;
        public List<double> weights = new List<double>();
        public List<double> inputs = new List<double>();

        /// <summary>
        /// This constructor initializes both the weight and bias with values between -0.5 and 0.5
        /// </summary>
        /// <param name="nInputs"></param>
        public Neuron(int nInputs)
        {
            var randomNumber = new Random();
            bias = randomNumber.NextDouble() - 0.5;

            numberInputs = nInputs;
            for (int i = 0; i < numberInputs; i++)
                weights.Add(randomNumber.NextDouble() - 0.5);
        }
    }


    /// <summary>
    /// The Layer class represents a list of artificial neurons.
    /// </summary>
    public class Layer
    {
        public int numberOfNeurons;
        public List<Neuron> neurons = new List<Neuron>();

        public Layer(int nNeurons, int numberOfInputPerNeuron)
        {
            numberOfNeurons = nNeurons;
            for (int i = 0; i < nNeurons; i++)
            {
                neurons.Add(new Neuron(numberOfInputPerNeuron));
            }
        }
    }

    public static class Utils
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
            {
                for (int h = 0; h < source.GetLength(1); h++)
                {
                    action(w, h);
                }
            }
        }
        /// <summary>
        /// Calculate a loss of a single sample
        /// </summary>
        /// <param name="predictions">Predicted array</param>
        /// <param name="desired_output">True value</param>
        /// <returns></returns>
        public static double calculate_loss(List<double> predictions, int[] desired_output) {
         
            double sum_loss = 0.0;
            for (int i = 0; i < desired_output.Length; ++i) {

              sum_loss += (predictions[i] - desired_output[i]) * (predictions[i] - desired_output[i]);

            }
            return sum_loss / 2;
           
        }

        public static int accuracy(List<double> predictions, int[] desired_output) {
            double maxPredicted = predictions.Max();
            int maxIndexPredicted = predictions.IndexOf(maxPredicted);

            double maxDesired = predictions.Max();
            int maxIndexDesired = predictions.IndexOf(maxDesired);

            return maxIndexDesired == maxIndexPredicted ? 1 : 0;
        }

        public static double SigmoidActivation(double value) 
        {
            double k = (double)Math.Exp(-1 * value);
            return 1 / (1.0 + k);
        }
        public static double SigmoidDerivation(double value)
        {

            return value * (1 - value);
        }

        public static void writeCSv(string path, List<Tuple<int, double, Tuple<double, double>, Tuple<double, double>>> result) {

            using (var writer = new StreamWriter(path))
            {
                // write header
                string heading = string.Format("{0},{1},{2},{3},{4},{5}", "Epoch,", "Learning_Rate,", "Training_Loss,", "Testing_Loss,", 
                    "Training_Accuracy,", "Testing_Accuracy" );
                writer.WriteLine(heading);
                writer.Flush();
                foreach ( var res in result)
                { 
                    var line = string.Format("{0},{1},{2},{3},{4},{5}", res.Item1,
                        res.Item2, res.Item3.Item1, res.Item4.Item1, res.Item3.Item2, res.Item4.Item2 );
                    writer.WriteLine(line);
                    writer.Flush();
                }
            }
        }
    }

    public class NeuralNetwork
    {

        public int numberInputs;
        public int numberOutputs;
        public int numberHidden;
        public int numberNeuronPerHidden;
        public double learning_rate;
        List<Layer> layers = new List<Layer>();

        public NeuralNetwork(int nI, int nO, int nH, int nPH)
        {
            numberInputs = nI;
            numberOutputs = nO;
            numberHidden = nH;
            numberNeuronPerHidden = nPH;
            

            if (numberHidden > 0)
            {
                layers.Add(new Layer(numberNeuronPerHidden, numberInputs));

                for (int i = 0; i < numberHidden - 1; i++)
                {
                    layers.Add(new Layer(numberNeuronPerHidden, numberNeuronPerHidden));
                }

                layers.Add(new Layer(numberOutputs, numberNeuronPerHidden));
            }
            else
            {
                layers.Add(new Layer(numberOutputs, numberInputs));
            }
        }

        private List<double> ForwardPass(List<double> inputs)
        {


            List<double> outputs = new List<double>();


            for (int i = 0; i < numberHidden + 1; i++)
            {
                if (i > 0)
                {
                    inputs = new List<double>(outputs);
                }
                outputs.Clear();

                for (int j = 0; j < layers[i].numberOfNeurons; j++)
                {
                    double Z = 0;
                    layers[i].neurons[j].inputs.Clear();

                    for (int k = 0; k < layers[i].neurons[j].numberInputs; k++)
                    {
                        layers[i].neurons[j].inputs.Add(inputs[k]);
                        Z += layers[i].neurons[j].weights[k] * inputs[k];
                    }

                    Z += layers[i].neurons[j].bias;
                    layers[i].neurons[j].output = Utils.SigmoidActivation(Z);

                    outputs.Add(layers[i].neurons[j].output);
                }
            }
            return outputs;
        }

        private void BackwardPass(List<double> outputs, int[] desiredOutput, double learning_rate)
        {
            double error = 0;
            for (int i = numberHidden; i >= 0; i--)
            {
                for (int j = 0; j < layers[i].numberOfNeurons; j++)
                {
                    if (i == numberHidden)
                    {
                        error = outputs[j] - desiredOutput[j];
                        layers[i].neurons[j].errorGradient = Utils.SigmoidDerivation(outputs[j]) * error;

                    }
                    else
                    {
                        layers[i].neurons[j].errorGradient = Utils.SigmoidDerivation(layers[i].neurons[j].output);
                        double errorGradSum = 0;
                        for (int p = 0; p < layers[i + 1].numberOfNeurons; p++)
                        {
                            errorGradSum += layers[i + 1].neurons[p].errorGradient * layers[i + 1].neurons[p].weights[j];
                        }
                        layers[i].neurons[j].errorGradient *= errorGradSum;
                    }
                    for (int k = 0; k < layers[i].neurons[j].numberInputs; k++)
                    {
                        if (i == numberHidden)
                        {
                            error = outputs[j] - desiredOutput[j];
                            layers[i].neurons[j].weights[k] -= learning_rate * layers[i].neurons[j].inputs[k] * error;
                        }
                        else
                        {
                            layers[i].neurons[j].weights[k] -= learning_rate * layers[i].neurons[j].inputs[k] * layers[i].neurons[j].errorGradient;
                        }
                    }
                    layers[i].neurons[j].bias -= learning_rate * layers[i].neurons[j].errorGradient;
                }

            }

        }

        public void Train(string imagesPath, string labelsPath, double learningRate)
        {
            foreach (var image in MnistReader.ReadData(imagesPath, labelsPath))
            {
                int[] desiredOutputs = new int[10];
                Byte[] inps = image.Data.Cast<Byte>().ToArray();
                List<double> predictions = new List<double>();
                List<double> inputs = inps.Select(i => (double)i / 255.0).ToList();
                //Console.WriteLine((int)image.Label);
                desiredOutputs[image.Label] = 1;
                predictions = ForwardPass(inputs);
                BackwardPass(predictions, desiredOutputs, learningRate);
                //Console.WriteLine("Desired outputs ");
                //foreach (var elt in desiredOutputs)
                //{
                //  Console.Write(" " + elt);
                //}
                //Console.WriteLine();
                //predictions = ForwardPass(inputs);


                //Console.WriteLine("Predictions");
                //foreach (var elt in predictions)
                //{
                  //  Console.Write(" " + elt);
                //}
                //BackwardPass(predictions, desiredOutputs);
            }



        }


        public Tuple<double, double> evaluate(string imagesPath, string labelsPath)
        {
            var accuracyValue = 0;
            var numberImage = 0;
            
            double loss = 0.0;
            
            foreach (var image in MnistReader.ReadData(imagesPath, labelsPath))
            {
                int[] desiredOuput = new int[10];
                desiredOuput[image.Label] = 1;
                numberImage += 1;
   
                Byte[] inps = image.Data.Cast<Byte>().ToArray();
                List<double> predictions = new List<double>();
                List<double> input = inps.Select(i => (double)i / 255.0).ToList();
                predictions = ForwardPass(input);
                //foreach (var elt in predictions)
                //{
                  //  Console.Write(" {0}", elt);
                //}
                //Console.WriteLine();
                var predictionValue = predictions.IndexOf(predictions.Max());
                var maxValue = predictions.Max();
                //Console.WriteLine("Expected {0} : predicted {1}: max value {2} ", image.Label, predictionValue, maxValue);
                if (predictionValue == image.Label)
                {
                    accuracyValue += 1;
                }
                
                loss += Utils.calculate_loss(predictions, desiredOuput);
            }
            double averageLoss = loss / numberImage;
            double averageAccuracy = (double)accuracyValue / numberImage;
 
            return Tuple.Create(averageLoss, averageAccuracy);

        }
    }


    class Program
    {
        static void Main(string[] args)
        {
            string result_file = @"\\Mac\Home\Desktop\mnist\result.csv";
            List<double> accuracyValues = new List<double>();
            double[] learningRates = { 0.2, 0.01, 0.1, 0.02, 0.3, 0.03, 0.4, 0.04};
            double learningRate = 0.1;
            // NeuralNetwork neuralnet = new NeuralNetwork(784, 10, 1, 45);
            Tuple<double, double> loss_accuracy;
            List<Tuple<int, double, Tuple<double, double>, Tuple<double, double>>> result = new List<Tuple<int, double, Tuple<double, double>, Tuple<double, double>>>();
            //double accuracy = 0;
            string format = "{0, -15}{1,-15}{2,-15}{3, -25}{4, -25}{5, -25}";
            string[] heading = new string[] { "Epoch", "Learning Rate", "Training Loss", "Testing Loss", "Training Accuracy", "Testing Accuracy" };
            Console.WriteLine(string.Format(format, heading));

            foreach (var learning_rate in learningRates) {
                NeuralNetwork neuralnet = new NeuralNetwork(784, 10, 1, 45);

                for (var epoch = 0; epoch < 10; ++epoch)
                {

                    neuralnet.Train(MnistReader.TrainImagesPath, MnistReader.TrainLabelsPath, learning_rate);
                    // evaluate on training data
                    var trainingMetric = neuralnet.evaluate(MnistReader.TrainImagesPath, MnistReader.TrainLabelsPath);

                    // evaluate on test data
                    var testingMetric = neuralnet.evaluate(MnistReader.TestImagesPath, MnistReader.TestLabelsPath);
                    result.Add(Tuple.Create(epoch, learning_rate, trainingMetric, testingMetric));

                    string[] row = new string[] { epoch.ToString(), learning_rate.ToString(), trainingMetric.Item1.ToString(), testingMetric.Item1.ToString(),
                trainingMetric.Item2.ToString(), testingMetric.Item2.ToString()};
                    Console.WriteLine(string.Format(format, row));
                    Utils.writeCSv(result_file, result);
                }
               

            }
        }
    }
}
    

