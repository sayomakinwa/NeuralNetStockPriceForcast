using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetStockPriceForcast
{
    class Firefly : IComparable<Firefly>
    {
        public double[] weight;
        public double error;
        public double intensity;

        public Firefly(int dim)
        {
            this.weight = new double[dim];
            this.error = 0.0;
            this.intensity = 0.0;
        }

        public int CompareTo(Firefly other)
        {
            // allow auto sort low error to high
            if (this.error < other.error)
                return -1;
            else if (this.error > other.error)
                return +1;
            else
                return 0;
        }
    }
}
