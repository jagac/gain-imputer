import unittest
from gain_imputer import GainImputerBuilder

class TestGainImputerBuilder(unittest.TestCase):
    def test_builder_with_dim(self):
        builder = GainImputerBuilder()
        builder.with_dim(10)
        self.assertEqual(builder.dim, 10)

    def test_builder_with_h_dim(self):
        builder = GainImputerBuilder()
        builder.with_h_dim(128)
        self.assertEqual(builder.h_dim, 128)

    def test_builder_with_cat_columns(self):
        builder = GainImputerBuilder()
        builder.with_cat_columns([0, 1, 2])
        self.assertEqual(builder.cat_columns, [0, 1, 2])

    def test_builder_with_batch_size(self):
        builder = GainImputerBuilder()
        builder.with_batch_size(256)
        self.assertEqual(builder.batch_size, 256)

    def test_builder_with_hint_rate(self):
        builder = GainImputerBuilder()
        builder.with_hint_rate(0.8)
        self.assertEqual(builder.hint_rate, 0.8)

    def test_builder_with_alpha(self):
        builder = GainImputerBuilder()
        builder.with_alpha(15)
        self.assertEqual(builder.alpha, 15)

    def test_builder_with_iterations(self):
        builder = GainImputerBuilder()
        builder.with_iterations(5000)
        self.assertEqual(builder.iterations, 5000)

if __name__ == "__main__":
    unittest.main()
