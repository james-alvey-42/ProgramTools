import unittest
from unittest.mock import patch
from functions import add, subtract, multiply, divide
from functions import Employee
import functions
import requests

class TestFunctions(unittest.TestCase):

	def test_add(self):
		self.assertEqual(add(10, 5), 15)
		self.assertEqual(add(-1, 1), 0)
		self.assertEqual(add(-1, -1), -2)

	def test_subtract(self):
		self.assertEqual(subtract(10, 5), 5)
		self.assertEqual(subtract(-1, 1), -2)
		self.assertEqual(subtract(-1, -1), 0)

	def test_multiply(self):
		self.assertEqual(multiply(10, 5), 50)
		self.assertEqual(multiply(-1, 1), -1)
		self.assertEqual(multiply(-1, -1), 1)

	def test_divide(self):
		self.assertEqual(divide(10, 5), 2)
		self.assertEqual(divide(-1, 1), -1)
		self.assertEqual(divide(-1, -1), 1)
		self.assertEqual(divide(5, 10), 0.5)
		with self.assertRaises(ValueError):
			divide(10, 0)

class TestEmployee(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		print('setUpClass')

	@classmethod
	def tearDownClass(cls):
		print('tearDownClass')

	def setUp(self):
		self.emp_1 = Employee('Corey', 'Schafer', 50000)
		self.emp_2 = Employee('Sue', 'Smith', 60000)

	def tearDown(self):
		pass

	def test_email(self):
		self.assertEqual(self.emp_1.email, 'Corey.Schafer@email.com')
		self.assertEqual(self.emp_2.email, 'Sue.Smith@email.com')

		self.emp_1.first = 'John'
		self.emp_2.first = 'Jane'

		self.assertEqual(self.emp_1.email, 'John.Schafer@email.com')
		self.assertEqual(self.emp_2.email, 'Jane.Smith@email.com')
