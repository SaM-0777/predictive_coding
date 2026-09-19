#include <iostream>
#include <vector>

int main()
{
  // Initial weights
  double w0 = 1.0;
  double w1 = 1.0;

  // Inputs and targets
  std::vector<double> inputs = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> targets = {2.0, 3.0, 4.0, 5.0, 6.0};

  // Hyperparameters
  int N_relax = 100;
  double step = 0.1;
  double lr = 0.1;
  int epochs = 500;

  // Declare outside the loop so they are accessible for the final Energy calculation
  double x0 = 0.0;
  double x1 = 0.0;
  double x2 = 0.0;

  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    for (size_t n = 0; n < inputs.size(); ++n)
    {
      x0 = inputs[n];
      x2 = targets[n];
      x1 = 0.0; // reset every example

      double e1 = 0.0;
      double e2 = 0.0;

      // Relaxation phase
      for (int t = 0; t < N_relax; ++t)
      {
        e1 = x1 - w0 * x0;
        e2 = x2 - w1 * x1;
        x1 = x1 + step * (-e1 + w1 * e2);
      }

      // Weight update phase
      e1 = x1 - w0 * x0;
      e2 = x2 - w1 * x1;
      w0 = w0 + lr * (e1 * x0);
      w1 = w1 + lr * (e2 * x1);
    }
  }

  // Energy on last example only (after last weight step)
  double term1 = x1 - w0 * x0;
  double term2 = x2 - w1 * x1;
  double E = 0.5 * (term1 * term1 + term2 * term2);

  // Print results
  std::cout << w0 << " " << w1 << " " << x1 << " " << E << std::endl;

  return 0;
}