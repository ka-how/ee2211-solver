"""Exact known-answer and edge-case checks for the dynamic event solver."""

from fractions import Fraction
import unittest

import sympy as sp

from ProbabilityUtils import format_probability_result, solve_probabilities


class ProbabilitySolverTests(unittest.TestCase):
    def answer(self, text, index=0):
        return solve_probabilities(text)["results"][index]

    def test_union_and_intersection_find_missing_marginal(self):
        answer = self.answer("a=.4; a*b=.1; a+b=.7; b=???")
        self.assertEqual(answer["status"], "solved")
        self.assertEqual(answer["value"], sp.Rational(2, 5))
        self.assertFalse(answer["undefined_possible"])

    def test_conditional_and_joint_find_denominator_and_reverse_bayes(self):
        answer = self.answer("P(A*B)=1/10; P(A|B)=1/4; P(B)=?")
        self.assertEqual(answer["value"], sp.Rational(2, 5))
        reverse = self.answer("a=.3; a|b=.5; b=.2; b|a=???")
        self.assertEqual(reverse["value"], sp.Rational(1, 3))

    def test_case_wrappers_percentages_comments_and_redundant_facts(self):
        answer = self.answer("""
        P(A) = 40% # Known marginal.
        A∩B = 1/10
        P(A∪B) = .7
        a|b = .25
        b = ???
        """)
        self.assertEqual(answer["value"], sp.Rational(2, 5))
        mapping = solve_probabilities({"a": Fraction(2,5), "a*b": "1/10", "a+b": "70%", "b": None})
        self.assertEqual(mapping["results"][0]["value"], sp.Rational(2,5))

    def test_complements_parentheses_and_multiple_queries(self):
        answers = solve_probabilities("a=.4; b=.5; a*b=.2; ~a=???; a'=???; ~(a+b)=???; a|~b=???")["results"]
        self.assertEqual([answer["value"] for answer in answers],
                         [sp.Rational(3,5), sp.Rational(3,5), sp.Rational(3,10), sp.Rational(2,5)])

    def test_no_independence_is_assumed_and_bounds_are_reported(self):
        answer = self.answer("a=.4; b=.5; a*b=???")
        self.assertEqual(answer["status"], "underdetermined")
        self.assertIsNone(answer["value"])
        self.assertEqual((answer["lower"], answer["upper"]), (0, sp.Rational(2,5)))
        self.assertTrue(answer["lower_inclusive"])
        self.assertTrue(answer["upper_inclusive"])
        self.assertIn("Not uniquely determined", format_probability_result(answer))

    def test_bounds_can_force_a_unique_answer_without_full_joint_identification(self):
        answer = self.answer("a=1; b=.4; a*b=???")
        self.assertEqual(answer["value"], sp.Rational(2,5))
        self.assertEqual(self.answer("a=0; a*b=???")["value"], 0)

    def test_impossible_or_conflicting_known_probabilities_are_rejected(self):
        for text in ("a=.4;a=.5;b=???", "a=.8;b=.8;a*b=.1;b=???",
                     "a=.4;a+b=.2;b=???", "a*b=.6;b=.4;a=???",
                     "a=.5;b=.4;a*b=.2;a|b=.8;b=???"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "inconsistent"):
                solve_probabilities(text)

    def test_known_conditionals_require_a_possible_positive_conditioning_event(self):
        for text in ("b=0;a|b=.5;a=???", "a=0;b|a=.5;b=???",
                     "a|b=1;b|a=0;a=???", "a|0=.5;b=???"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "conditioning event"):
                solve_probabilities(text)

    def test_zero_probability_conditionals_are_undefined(self):
        answer = self.answer("b=0;a|b=???")
        self.assertEqual(answer["status"], "undefined")
        self.assertIsNone(answer["value"])
        self.assertIn("Undefined", format_probability_result(answer))

    def test_potentially_zero_conditioning_events_are_disclosed(self):
        answer = self.answer("a=0;a|b=???")
        self.assertEqual(answer["value"], 0)
        self.assertTrue(answer["undefined_possible"])
        answer = self.answer("a|a=???")
        self.assertEqual(answer["value"], 1)
        self.assertTrue(answer["undefined_possible"])
        defined = self.answer("b=.5;a|a=1;a|b=???")
        self.assertFalse(defined["undefined_possible"])

    def test_open_bounds_from_known_conditionals(self):
        answers = solve_probabilities("a|b=.5;a=???;b=???")["results"]
        a, b = answers
        self.assertEqual((a["lower"],a["upper"]), (0,1))
        self.assertFalse(a["lower_inclusive"])
        self.assertFalse(a["upper_inclusive"])
        self.assertFalse(b["lower_inclusive"])
        self.assertTrue(b["upper_inclusive"])
        self.assertIn("(0, 1]", format_probability_result(b))

    def test_tautologies_and_empty_events(self):
        answers = solve_probabilities("a+~a=1;a*~a=0; a+~a=???;a*~a=???")["results"]
        self.assertEqual([answer["value"] for answer in answers], [1,0])

    def test_related_answers_and_known_only_input(self):
        result = solve_probabilities("a=.4;a*b=.1;a+b=.7;b=???", include_related=True)
        self.assertGreater(len(result["related"]), 0)
        by_name = {answer["expression"]: answer for answer in result["related"]}
        self.assertEqual(by_name["P(A|B)"]["value"], sp.Rational(1,4))
        self.assertGreater(len(solve_probabilities("a=.4")["results"]), 0)

    def test_invalid_syntax_and_values_are_not_evaluated(self):
        for text in ("a=-.1;b=?", "a=1.1;b=?", "a=nan;b=?", "a=1/0;b=?",
                     "a == .4", "a=.4=.5", "d=.2;b=?", "a/b=.2;b=?", "a|b|a=.2",
                     "a=(1+2);b=?", "__import__('os')=.2;b=?", "a**b=.2", ""):
            with self.subTest(text=text), self.assertRaises(ValueError):
                solve_probabilities(text)

    def test_many_joint_distributions_give_exact_conditional_answers(self):
        for weights in ((1,2,3,4), (0,1,2,3), (5,0,1,0), (1,1,0,0), (3,7,11,13)):
            joint = [sp.Rational(value,sum(weights)) for value in weights]
            ab, a_only, b_only, neither = joint
            a, b = ab+a_only, ab+b_only
            result = solve_probabilities({"a":a, "b":b, "a*b":ab, "a|b":"?", "b|a":"?", "~(a+b)":"?"})
            answers = result["results"]
            if b:
                self.assertEqual(answers[0]["value"], ab/b)
            else:
                self.assertEqual(answers[0]["status"], "undefined")
            if a:
                self.assertEqual(answers[1]["value"], ab/a)
            self.assertEqual(answers[2]["value"], neither)

    def test_c_can_be_known_or_unknown_and_names_are_case_insensitive(self):
        answer = self.answer("a*c=.12; a|c=.3; c=???")
        self.assertEqual(answer["value"], sp.Rational(2,5))
        result = solve_probabilities("P(C)=40%; C'=???; A=???", include_related=True)
        self.assertEqual(result["events"], ["a", "b", "c"])
        self.assertEqual(result["results"][0]["value"], sp.Rational(3,5))
        self.assertEqual(result["results"][1]["status"], "underdetermined")
        self.assertTrue(all(len(vertex) == 8 for vertex in result["joint_vertices"]))
        self.assertIn("P(C)", {item["expression"] for item in result["related"]})

    def test_three_event_union_can_solve_c_and_triple_intersection(self):
        known = "a=.4;b=.5;a*b=.2;a*c=.1;b*c=.15; a*b*c=.05;a+b+c=.75; c=???"
        self.assertEqual(self.answer(known)["value"], sp.Rational(1,4))
        triple = "a=.4;b=.5;c=.25;a*b=.2;a*c=.1;b*c=.15;a+b+c=.75; a*b*c=???"
        self.assertEqual(self.answer(triple)["value"], sp.Rational(1,20))

    def test_three_event_conditionals_and_complements(self):
        known = "a=.4;b=.5;c=.25;a*b=.2;a*c=.1;b*c=.15;a*b*c=.05;"
        queries = "a|(b*c)=???;(a*b)|c=???;a|(b+c)=???;P(C'|A*B)=???;~(a+b+c)=???"
        answers = solve_probabilities(known + queries)["results"]
        self.assertEqual([item["value"] for item in answers],
                         [sp.Rational(1,3),sp.Rational(1,5),sp.Rational(5,12),sp.Rational(3,4),sp.Rational(1,4)])

    def test_three_event_pairwise_independence_does_not_imply_mutual_independence(self):
        answer = self.answer("a=.5;b=.5;c=.5;a*b=.25;a*c=.25;b*c=.25; a*b*c=???")
        self.assertEqual(answer["status"], "underdetermined")
        self.assertEqual((answer["lower"],answer["upper"]), (0,sp.Rational(1,4)))

    def test_three_event_global_consistency_is_checked(self):
        # Each pair can be disjoint with these marginals, but all three cannot.
        for text in ("a=.5;b=.5;c=.5;a*b=0;a*c=0;b*c=0;c=???",
                     "a*b=.1;a*b*c=.2;c=???"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "inconsistent"):
                solve_probabilities(text)

    def test_three_event_zero_conditionals_and_open_bounds(self):
        answer = self.answer("b*c=0; a|(b*c)=???")
        self.assertEqual(answer["status"], "undefined")
        with self.assertRaisesRegex(ValueError, "conditioning event"):
            solve_probabilities("b*c=0; a|(b*c)=.5; c=???")
        conditional = self.answer("a|c=.5; c=???")
        self.assertEqual((conditional["lower"],conditional["upper"]), (0,1))
        self.assertFalse(conditional["lower_inclusive"])
        self.assertTrue(conditional["upper_inclusive"])


if __name__ == "__main__":
    unittest.main()
