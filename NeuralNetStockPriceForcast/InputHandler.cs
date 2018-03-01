using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetStockPriceForcast
{
    class InputOutputHandler
    {
        int dim, numRecs;
        string header, inputDir, outputDir, filename;
        public InputOutputHandler(int dimension = 4, int records = 20, string file = "alk.csv")
        {
            this.dim = dimension;
            this.numRecs = records;
            this.inputDir = "data/";
            this.outputDir = "data/output/";
            this.filename = file;
            string[] head = file.Split('.');
            this.header = head[0].ToUpper();
        }
        public double[][][] GetInputs() //training data
        {
            string training = System.IO.File.ReadAllText(this.inputDir + "training/" + this.filename);
            training = training.Replace('\n', '\r');

            string[] lines = training.Split(new char[] { '\r' }, StringSplitOptions.RemoveEmptyEntries);

            double[][][] values = new double[this.numRecs][][];
            int i = 1;

            for (int r = 0; r < this.numRecs; r++)
            {
                double[] temp = new double[this.dim];
                for (int c = 0; c < this.dim; c++)
                {
                    string[] lineR = lines[i++].Split(',');
                    temp[c] = Convert.ToDouble(lineR[0]);
                    //Console.Write(lineR[0]+", ");
                }
                string[] lineR2 = lines[i++].Split(',');
                values[r] = new double[2][] { temp, new double[] { Convert.ToDouble(lineR2[0]) } };
                //Console.WriteLine(lineR2[0]+" ==== "+(r+1).ToString());
                //Console.ReadLine();

            }

            /*double[][][] trainingData = new double[][][] {
                new double[][] {new double[] {7.22, 7.4, 7.33, 7.29}, new double[] {7.38}},
                new double[][] {new double[] {7.34, 7.5, 7.76, 7.52}, new double[] {7.49}},
                new double[][] {new double[] {7.39, 7.5, 7.88, 7.8}, new double[] {8.34}},
                new double[][] {new double[] {8.21, 8.28, 8.42, 8.22}, new double[] {8.3}},
                new double[][] {new double[] {8.57, 8.64, 8.54, 8.39}, new double[] {8.51}},
                new double[][] {new double[] {8.83, 8.77, 8.74, 8.58}, new double[] {8.82}},
                new double[][] {new double[] {8.82, 8.74, 8.79, 8.81}, new double[] {9.12}},
                new double[][] {new double[] {9.33, 9.25, 9.33, 9.34}, new double[] {9.18}},
                new double[][] {new double[] {9.11, 9.14, 8.78, 8.82}, new double[] {9.04}},
                new double[][] {new double[] {9.19, 8.9, 8.89, 8.88}, new double[] {8.91}}
            };
            return trainingData;*/
            return values;
        }

        public double[][][] GetTestingData()
        {
            string testing = System.IO.File.ReadAllText(this.inputDir + "testing/" + this.filename);
            testing = testing.Replace('\n', '\r');
            string[] lines = testing.Split(new char[] { '\r' }, StringSplitOptions.RemoveEmptyEntries);

            double[][][] values = new double[this.numRecs][][];
            int i = 0;

            string[] lineRow = lines[i++].Split(',');
            this.header = lineRow[0];

            for (int r = 0; r < this.numRecs; r++)
            {
                double[] temp = new double[this.dim];
                for (int c = 0; c < this.dim; c++)
                {
                    string[] lineR = lines[i++].Split(',');
                    temp[c] = Convert.ToDouble(lineR[0]);
                }
                string[] lineR2 = lines[i++].Split(',');
                values[r] = new double[2][] { temp, new double[] { Convert.ToDouble(lineR2[0]) } };

            }
            return values;
        }

        public void ShowVector(string[] v, bool nl)
        {
            for (int i = 0; i < v.Length; ++i)
                Console.Write(v[i] + " ");
            if (nl == true)
                Console.WriteLine("");
        }

        public void WriteOutputs(double[] weights) //writes only output from testing data
        {
            string[][] csvData = new string[this.numRecs + 4][];
            for (int s = 0; s < csvData.Length; s++)
            {
                csvData[s] = new string[this.dim + 4];
                for (int d = 0; d < this.dim + 4; d++)
                {
                    csvData[s][d] = ".";
                }
            }
            
            csvData[0][0] = this.header;

            csvData[1][0] = "Weights:";
            Console.WriteLine("--------------------");
            ShowVector(csvData[0], true);
            ShowVector(csvData[1], true);

            for (int d = 0; d < weights.Length; d++)
            {
                csvData[2][d] = Convert.ToString(weights[d]);
            }
            
            Console.WriteLine("--------------------");
            ShowVector(csvData[0], true);
            ShowVector(csvData[1], true);
            ShowVector(csvData[2], true);

            for (int d = 0; d < weights.Length; d++)
            {
                csvData[3][d] = "t" + d.ToString();
            }
            csvData[3][(this.dim+4) - 3] = "Predicted";
            csvData[3][(this.dim + 4) - 2] = "Actual";
            csvData[3][(this.dim + 4) - 1] = "Error";
            
            Console.WriteLine("--------------------");
            ShowVector(csvData[0], true);
            ShowVector(csvData[1], true);
            ShowVector(csvData[2], true);
            ShowVector(csvData[3], true);

            double[][][] testingData = this.GetTestingData();
            
            ANNPerceptron res = new ANNPerceptron();
            for (int d = 0; d < testingData.Length; d++)
            {
                double activation = res.Result(testingData[d][0], weights);
                int x;
                for (x = 0; x < this.dim; x++)
                {
                    csvData[d + 4][x] = Convert.ToString(testingData[d][0][x]);
                }
                x++;
                csvData[d + 4][x++] = (activation).ToString("F2");
                csvData[d + 4][x++] = (testingData[d][1][0]).ToString();
                csvData[d + 4][x] = (Math.Abs(activation - testingData[d][1][0])).ToString("F2");
                
                Console.Write("csvData[{0}] ", d + 4);
                ShowVector(csvData[d + 4], true);
            }

            Console.WriteLine("===================================");

            StringBuilder sb = new StringBuilder();
            for (int index = 0; index < csvData.Length; index++)
            {
                ShowVector(csvData[index], true);
                sb.AppendLine(string.Join(",", csvData[index]));
            }
            System.IO.File.WriteAllText(this.outputDir + this.filename, sb.ToString());

        }
    }
}
