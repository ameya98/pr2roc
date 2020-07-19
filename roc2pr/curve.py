from __future__ import division
import numpy as np

# A generic curve.
class Curve:
    def __init__(self, points, label=None):
        try:
            points = list(points)
        except (ValueError, TypeError):
            raise ValueError('Could not convert points to list.')

        if len(points) <= 1:
            raise ValueError('Need more than one point for a curve.')

        try:
            # Sort by x-value.
            points = sorted(np.array(points), key=lambda p: p[0])
            self.x_vals = np.array([point[0] for point in points])
            self.y_vals = np.array([point[1] for point in points])
        except (IndexError, TypeError):
            raise ValueError('Points must be passed as a sequence of 2-tuples.')

        self.label = label

    # Resamples this curve along equally spaced x-values.
    def resample(self, num_points):        
        x_sampled = np.linspace(self.x_vals[0], self.x_vals[-1], num_points)
        y_sampled = np.zeros(num_points)

        next_index = 1
        prev_x = self.x_vals[0]
        prev_y = self.y_vals[0]
        next_x = self.x_vals[1]
        next_y = self.y_vals[1]

        for index, x in enumerate(x_sampled):
            while x > next_x:
                next_index += 1
            
                prev_x = self.x_vals[next_index - 1]
                prev_y = self.y_vals[next_index - 1]
                next_x = self.x_vals[next_index]
                next_y = self.y_vals[next_index]

            y_sampled[index] = prev_y + (x - prev_x)*(next_y - prev_y)/(next_x - prev_x)

        points_sampled = zip(x_sampled, y_sampled)
        return Curve(points_sampled)
        
    # Returns the points.
    def points(self):
        return list(zip(self.x_vals, self.y_vals))