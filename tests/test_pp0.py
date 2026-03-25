import unittest

from pushpop.pp0 import PP0ExecutionError, execute


class PP0InterpreterTests(unittest.TestCase):
    def test_executes_simple_add_program(self) -> None:
        result = execute("2 3 ADD END")

        self.assertEqual(result.final_stack, (5,))
        self.assertEqual(result.final_top, 5)
        self.assertEqual([step.token for step in result.trace], ["2", "3", "ADD", "END"])

    def test_trace_rows_capture_stack_before_and_after_each_step(self) -> None:
        result = execute("3 1 4 SWAP SUB END")

        self.assertEqual(
            result.trace_rows(),
            [
                {
                    "step": 0,
                    "pc": 0,
                    "token": "3",
                    "stack_before": (),
                    "stack_after": (3,),
                    "depth_before": 0,
                    "depth_after": 1,
                    "top_before": None,
                    "top_after": 3,
                },
                {
                    "step": 1,
                    "pc": 1,
                    "token": "1",
                    "stack_before": (3,),
                    "stack_after": (3, 1),
                    "depth_before": 1,
                    "depth_after": 2,
                    "top_before": 3,
                    "top_after": 1,
                },
                {
                    "step": 2,
                    "pc": 2,
                    "token": "4",
                    "stack_before": (3, 1),
                    "stack_after": (3, 1, 4),
                    "depth_before": 2,
                    "depth_after": 3,
                    "top_before": 1,
                    "top_after": 4,
                },
                {
                    "step": 3,
                    "pc": 3,
                    "token": "SWAP",
                    "stack_before": (3, 1, 4),
                    "stack_after": (3, 4, 1),
                    "depth_before": 3,
                    "depth_after": 3,
                    "top_before": 4,
                    "top_after": 1,
                },
                {
                    "step": 4,
                    "pc": 4,
                    "token": "SUB",
                    "stack_before": (3, 4, 1),
                    "stack_after": (3, 3),
                    "depth_before": 3,
                    "depth_after": 2,
                    "top_before": 1,
                    "top_after": 3,
                },
                {
                    "step": 5,
                    "pc": 5,
                    "token": "END",
                    "stack_before": (3, 3),
                    "stack_after": (3, 3),
                    "depth_before": 2,
                    "depth_after": 2,
                    "top_before": 3,
                    "top_after": 3,
                },
            ],
        )

    def test_accepts_sequence_input_and_returns_modular_result(self) -> None:
        result = execute(["2", "9", "SUB", "END"])

        self.assertEqual(result.final_stack, (3,))
        self.assertEqual(result.final_top, 3)

    def test_executes_structural_program_example(self) -> None:
        result = execute("5 2 DUP ADD END")

        self.assertEqual(result.final_stack, (5, 4))
        self.assertEqual(result.final_top, 4)

    def test_raises_on_stack_underflow(self) -> None:
        with self.assertRaisesRegex(
            PP0ExecutionError,
            r"ADD requires stack depth >= 2 at pc 0, got 0",
        ):
            execute("ADD END")

    def test_requires_end_to_be_final(self) -> None:
        with self.assertRaisesRegex(PP0ExecutionError, r"END may only appear as the final token"):
            execute("2 END 3")


if __name__ == "__main__":
    unittest.main()
