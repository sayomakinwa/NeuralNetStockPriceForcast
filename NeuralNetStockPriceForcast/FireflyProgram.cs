using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetStockPriceForcast
{
    class FireflyProgram
    {
        static double aVar;
        static void Main(string[] args)
        {
            Console.Write("Input number of fireflies (between 15 and 40): ");
            int numFireflies = Convert.ToInt16(Console.ReadLine()); // typically 15-40
            if (numFireflies < 15 || numFireflies > 40) numFireflies = 40;

            Console.Write("Input the dimensionality (typically 4): ");
            int dim = Convert.ToInt16(Console.ReadLine());
            if (dim < 3 || dim > 5) dim = 4;

            Console.Write("Input the maximum no of epochs (typically 1000): ");
            int maxEpochs = Convert.ToInt16(Console.ReadLine());
            if (maxEpochs < 500 || maxEpochs > 2000) maxEpochs = 1000;

            Console.Write("Input value for the variable in the value of alpha: ");
            FireflyProgram.aVar = Convert.ToInt16(Console.ReadLine());
            if (FireflyProgram.aVar < 0.0 || FireflyProgram.aVar > 2.0) FireflyProgram.aVar = 0.98;

            int seed = 0;

            Console.Write("Input no of training/testing records to use (typically between 10 and 50): ");
            int numData = Convert.ToInt16(Console.ReadLine());
            if (numData < 5 || numData > 100) numData = 50;

            Console.Write("input the name of your training/testing file (eg. alk.csv): ");
            string filename = Console.ReadLine();
            if (filename == "") filename = "alk.csv";

            Console.Clear();

            Console.WriteLine("\nSetting numFireflies        = " + numFireflies);
            Console.WriteLine("Setting problem dim         = " + dim);
            Console.WriteLine("Setting maxEpochs           = " + maxEpochs);
            Console.WriteLine("Setting initialization seed = " + seed);
            Console.WriteLine("Setting no training/testing data = " + numData);
            Console.WriteLine("Setting no training/testing file name = " + filename);

            Console.WriteLine("\nStarting firefly algorithm\n");
            double[] bestWeights = Solve(numFireflies, dim, seed, maxEpochs, numData, filename);
            Console.WriteLine("\nFinished\n");

            Console.WriteLine("Best solution found: ");
            Console.Write("weights = ");
            ShowVector(bestWeights, 4, true);
            
            double error = Error(bestWeights, dim, numData, filename);
            Console.Write("Error at best weight = ");
            Console.WriteLine(error.ToString("F4"));
            //DisplayActivation(bestWeights, dim, numData, filename);
            InputOutputHandler myInput = new InputOutputHandler(dim, numData, filename);
            myInput.WriteOutputs(bestWeights);

            Console.WriteLine("\nEnd of algorithm\n");
            Console.ReadLine();
        }
        static void ShowVector(double[] v, int dec, bool nl)
        {
            for (int i = 0; i < v.Length; ++i)
                Console.Write(v[i].ToString("F" + dec) + " ");
            if (nl == true)
                Console.WriteLine("");
        }

        static double[] Solve(int numFireflies, int dim, int seed, int maxEpochs, int numData, string filename)
        {
            Random rnd = new Random(seed);
            double minWeight = 0.0; // specific to Result function
            double maxWeight = 1.0;

            double B0 = 1.0;  // beta (attractiveness base)
            //double betaMin = 0.20;
            double g = 1.0;   // gamma (absorption for attraction)
            //double a = 0.20;    // alpha
            double a0 = 1.0;    // base alpha for decay
            int displayInterval = maxEpochs / 10;

            double bestError = double.MaxValue;
            double[] bestWeights = new double[dim]; // best ever

            Firefly[] swarm = new Firefly[numFireflies]; // all null

            // initialize swarm at random weights
            for (int i = 0; i < numFireflies; ++i)
            {
                swarm[i] = new Firefly(dim); // weight 0, error and intensity 0.0
                for (int k = 0; k < dim; ++k) // random weight
                    swarm[i].weight[k] = (maxWeight - minWeight) * rnd.NextDouble() + minWeight;
                swarm[i].error = Error(swarm[i].weight, dim, numData, filename); // this function error will call the ANN code
                swarm[i].intensity = 1 / (swarm[i].error + 1); // +1 prevent div by 0
                
                
                
                //coming back here...



                Console.Write("Firefly "+i.ToString()+": ");
                ShowVector(swarm[i].weight, 4, false);
                Console.WriteLine(" error " + swarm[i].error.ToString() + " intensity: " + swarm[i].intensity.ToString());

                if (swarm[i].error < bestError)
                {
                    bestError = swarm[i].error;
                    for (int k = 0; k < dim; ++k)
                        bestWeights[k] = swarm[i].weight[k];
                }
            }

            int epoch = 0;
            while (epoch < maxEpochs) // main processing
            {
                //if (bestError < errThresh) break; // are we good?
                if (epoch % displayInterval == 0 && epoch < maxEpochs) // show progress?
                {
                    string sEpoch = epoch.ToString().PadLeft(6);
                    Console.Write("epoch = " + sEpoch);
                    Console.WriteLine("   error = " + bestError.ToString("F14"));
                }

                for (int i = 0; i < numFireflies; ++i) // each firefly
                {
                    for (int j = 0; j < numFireflies; ++j) // each other firefly. weird!
                    {
                        if (swarm[i].intensity < swarm[j].intensity)
                        { //this defines the move which learning rate in ANN is a function of
                            // curr firefly i is less intense (i is worse) so move i toward j
                            double r = Distance(swarm[i].weight, swarm[j].weight);
                            double beta = B0 * Math.Exp(-g * r * r); // original 
                            //double beta = (B0 - betaMin) * Math.Exp(-g * r * r) + betaMin; // better
                            //double a = a0 * Math.Pow(0.98, epoch); // better
                            double a = a0 * Math.Pow(FireflyProgram.aVar, epoch); // better
                            for (int k = 0; k < dim; ++k)
                            {
                                swarm[i].weight[k] += beta * (swarm[j].weight[k] - swarm[i].weight[k]);
                                swarm[i].weight[k] += a * (rnd.NextDouble() - 0.5);
                                if (swarm[i].weight[k] < minWeight) swarm[i].weight[k] = (maxWeight - minWeight) * rnd.NextDouble() + minWeight;
                                if (swarm[i].weight[k] > maxWeight) swarm[i].weight[k] = (maxWeight - minWeight) * rnd.NextDouble() + minWeight;
                            }
                            swarm[i].error = Error(swarm[i].weight, dim, numData, filename);
                            swarm[i].intensity = 1 / (swarm[i].error + 1);
                        }
                    } // j
                } // i each firefly

                Array.Sort(swarm); // low error to high
                if (swarm[0].error < bestError) // new best?
                {
                    bestError = swarm[0].error;
                    for (int k = 0; k < dim; ++k)
                        bestWeights[k] = swarm[0].weight[k];
                }
                ++epoch;
            } // while
            return bestWeights;
        } // Solve
        static double Distance(double[] posA, double[] posB)
        {
            double ssd = 0.0; // sum squared diffrences (Euclidean)
            for (int i = 0; i < posA.Length; ++i)
                ssd += (posA[i] - posB[i]) * (posA[i] - posB[i]);
            return Math.Sqrt(ssd);
        }

        static void DisplayActivation(double[] weights, int dim, int numData, string filename)
        {
            InputOutputHandler myInput = new InputOutputHandler(dim, numData, filename);
            double[][][] data = myInput.GetInputs();
            ANNPerceptron res = new ANNPerceptron();

            for (int i = 0; i < data.Length; i++)
            {
                double activation = res.Result(data[i][0], weights);
                //Console.WriteLine("Activation= " + activation.ToString() + " || actual sol= " + data[i][1][0].ToString() + " || single error= " + (activation - data[i][1][0]).ToString());
            }
        }
        static double Error(double[] weights, int dim, int numData, string filename)
        {
            InputOutputHandler myInput = new InputOutputHandler(dim, numData, filename);
            double[][][] data = myInput.GetInputs();
            ANNPerceptron res = new ANNPerceptron();
            double sumSq = 0;

            for (int i = 0; i < data.Length; i++) 
            {
                double activation = res.Result(data[i][0], weights);
                //Console.WriteLine(" - activation= " + activation.ToString() + " - actual sol= " + data[i][1][0].ToString());
                
                sumSq += Math.Pow(activation - data[i][1][0], 2);
            }
            double err = Math.Sqrt((1.0 / data.Length) * sumSq);
            return err;
        }
    }
}
  