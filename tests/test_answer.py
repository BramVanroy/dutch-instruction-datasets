import shutil

import pandas as pd
import pytest

from dutch_data.processor import AnswerGenerator


class TestAnswerGenerator:
    def test_answer_generator(self, text_generator):
        answer_generator = AnswerGenerator(
            dataset_name="BramVanroy/test-dataset-dont-delete",
            text_generator=text_generator,
            user_column="question",
            system_column="system",
            response_column="response",
            max_samples=4,
            dout="answer_test_output"
        )
        answer_generator.process_dataset()

        pfout = answer_generator.dout / "tmp_openai_done.jsonl"

        assert pfout.exists()
        assert pfout.is_file()
        assert pfout.stat().st_size > 0

        done_data = pd.read_json(pfout, lines=True)

        assert len(done_data.index) == 4
        assert answer_generator.response_column in done_data.columns

        shutil.rmtree(answer_generator.dout)

    def test_invalid_arguments(self, hf_generator):
        # Only test with hfgenerator
        # No user column specified --
        with pytest.raises(ValueError):
            answer_generator = AnswerGenerator(
                dataset_name="BramVanroy/test-dataset-dont-delete",
                text_generator=hf_generator,
                response_column="response",
                dout="answer_test_output"
            )

        with pytest.raises(ValueError):
            # 'chosen' response column already in dataset
            answer_generator = AnswerGenerator(
                dataset_name="BramVanroy/test-dataset-dont-delete",
                text_generator=hf_generator,
                response_column="chosen",
                dout="answer_test_output"
            )