using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetStockPriceForcast
{
    class ANNPerceptron
    {
        public double Result(double[] data, double[] weights)
        {
            //InputHandler myInput = new InputHandler();
            //double[][][] data = myInput.GetInputs();

            //System.out.println("Starting weights: " + Arrays.toString(weights));
            // Calculate weighted input
            double weightedSum = 0;
            for(int i=0; i < data.Length; i++) {
                weightedSum += data[i] * weights[i];
            }
            return weightedSum;
        }
    }
}
