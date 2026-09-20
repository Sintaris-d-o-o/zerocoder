"""Тесты сервиса прескоринга без обращения к модели (задание 10.7)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gpt_service  # noqa: E402
import normalize  # noqa: E402
import parser as resume_parser  # noqa: E402
import scoring  # noqa: E402
from models import Assessment, verdict_for  # noqa: E402


VACANCY = """Специалист по регуляторным вопросам.
Требования: опыт работы с MDR от 3 лет, знание ISO 13485,
опыт прохождения аудита нотифицированного органа, английский B2."""

RESUME_GOOD = """Регуляторный специалист, 5 лет опыта.
Вела технические файлы по Регламенту 2017/745, проходила аудит нотифицированного органа,
поддерживала систему менеджмента качества по ISO 13485. Английский B2, коммуникация
с BSI. Управление рисками по ISO 14971."""

RESUME_OFFTOPIC = """Frontend-разработчик, 6 лет опыта.
React, TypeScript, Next.js. Разработка интерфейсов, работа в команде, английский B1."""


# --------------------------------------------------------------- нормализация

def test_synonyms_are_folded():
    """«Регламент 2017/745» и MDR — одно и то же требование."""
    assert "mdr" in normalize.normalize_text("Опыт работы с Регламентом 2017/745")
    assert "mdr" in normalize.normalize_text("Знание MDR")
    assert "iso 13485" in normalize.normalize_text("Поддержка ИСО 13485")


def test_long_synonym_wins_over_short_one():
    """«regulation 2017/745» не должен схлопнуться по куску «2017/745»."""
    assert normalize.normalize_text("Regulation 2017/745 compliance") == "mdr compliance"


def test_stop_words_do_not_count_as_skills():
    """Иначе «требования» и «опыт» из вакансии засчитывались бы как совпадение."""
    found = normalize.tokens("Требования: опыт работы, знание MDR")
    assert "mdr" in found
    assert "требования" not in found and "опыт" not in found


def test_years_take_the_maximum_not_the_sum():
    """«3 года в X и 2 года в Y» — это не пять лет стажа."""
    assert normalize.years_of_experience("3 года в одной компании, 2 года в другой") == 3.0
    assert normalize.years_of_experience("Опыт 5 лет") == 5.0
    assert normalize.years_of_experience("Без указания") == 0.0


def test_seniority_bands_match_the_n3_agent():
    assert normalize.seniority(0.5) == "junior"
    assert normalize.seniority(3) == "middle"
    assert normalize.seniority(6) == "senior"
    assert normalize.seniority(12) == "lead"


def test_position_family_counts_hits_not_first_match():
    """В любом резюме есть слово «разработка» — по первому совпадению
    регуляторщик оказался бы программистом."""
    assert normalize.position_family(VACANCY) == "regulatory"
    assert normalize.position_family(RESUME_GOOD) == "regulatory"
    assert normalize.position_family(RESUME_OFFTOPIC) == "software"


# --------------------------------------------------------------- разбор PDF

def test_hyphen_break_is_glued_before_line_break():
    """Порядок важен: иначе дефис останется посреди слова."""
    assert resume_parser.clean_text("регу-\nляторный") == "регуляторный"


def test_line_breaks_inside_a_paragraph_disappear():
    text = "Вела технические файлы\nпо MDR и проходила аудит."
    assert "\n" not in resume_parser.clean_text(text)


def test_line_break_after_sentence_is_kept_as_separator():
    cleaned = resume_parser.clean_text("Первое предложение.\n\nВторое предложение.")
    assert "\n\n" in cleaned


def test_bullets_keep_their_own_lines():
    cleaned = resume_parser.clean_text("Задачи\n- первая\n- вторая")
    assert cleaned.count("\n") >= 2


def test_limit_cuts_on_a_boundary_and_says_so():
    long_text = ("Абзац про регуляторику. " * 400)
    cut = resume_parser.limit(long_text, 1000)
    assert len(cut) <= 1000 + 40
    assert "обрезано" in cut


def test_short_text_is_untouched():
    assert resume_parser.limit("коротко", 6000) == "коротко"


# --------------------------------------------------------------- веса и формула

def test_weights_are_normalized_to_one():
    """Пользователь двигает три ползунка, сумма почти никогда не равна 100."""
    w = scoring.normalize_weights({"hard": 70, "experience": 20, "soft": 20})
    assert pytest.approx(sum(w.values()), abs=1e-9) == 1.0
    assert w["hard"] > w["experience"] > 0


def test_zero_weights_fall_back_to_defaults():
    assert scoring.normalize_weights({"hard": 0, "experience": 0, "soft": 0}) \
        == scoring.DEFAULT_WEIGHTS


def test_relevant_resume_scores_higher_than_offtopic_one():
    weights = scoring.DEFAULT_WEIGHTS
    good = scoring.heuristic_score(scoring.analyse(VACANCY, RESUME_GOOD), weights)
    off = scoring.heuristic_score(scoring.analyse(VACANCY, RESUME_OFFTOPIC), weights)
    assert good > off + 20, (good, off)


def test_family_match_moves_the_score_both_ways():
    """Поправка за семейство должностей — то, ради чего она взята из N3:
    без неё богатое, но чужое резюме обгоняет профильное."""
    relevant = scoring.analyse(VACANCY, RESUME_GOOD)
    foreign = scoring.analyse(VACANCY, RESUME_OFFTOPIC)
    assert relevant.family_match is True
    assert foreign.family_match is False


def test_experience_stops_counting_after_the_cap():
    at_cap, _ = scoring.experience_score("Опыт 15 лет")
    way_over, _ = scoring.experience_score("Опыт 30 лет")
    assert at_cap == way_over == 100.0


def test_blend_respects_the_share():
    assert scoring.blend(100, 0, 1.0) == 100
    assert scoring.blend(100, 0, 0.0) == 0
    assert scoring.blend(80, 40, 0.5) == 60


def test_verdict_thresholds_match_the_n3_agent():
    assert verdict_for(60) == "подходит"
    assert verdict_for(59) == "почти подходит"
    assert verdict_for(45) == "почти подходит"
    assert verdict_for(44) == "не подходит"


def test_score_never_leaves_the_scale():
    breakdown = scoring.analyse(VACANCY, RESUME_GOOD)
    assert 0 <= scoring.heuristic_score(breakdown, {"hard": 100}) <= 100


# --------------------------------------------------------------- ремонт JSON

def test_plain_json_is_parsed():
    assert gpt_service.repair_json('{"score": 70}') == {"score": 70}


def test_code_block_wrapper_is_stripped():
    raw = '```json\n{"score": 70, "summary": "ок"}\n```'
    assert gpt_service.repair_json(raw)["score"] == 70


def test_text_around_the_object_is_ignored():
    raw = 'Вот результат:\n{"score": 55, "summary": "почти"}\nГотово!'
    assert gpt_service.repair_json(raw)["score"] == 55


def test_braces_inside_strings_do_not_break_extraction():
    """Скобки считаются, а не ищутся с конца — иначе строка со скобкой ломала бы разбор."""
    raw = '{"summary": "формула {a+b} в тексте", "score": 50}'
    assert gpt_service.repair_json(raw)["score"] == 50


def test_trailing_comma_and_single_quotes_are_repaired():
    raw = "{'score': 42, 'summary': 'почти',}"
    assert gpt_service.repair_json(raw)["score"] == 42


def test_unrepairable_answer_returns_empty_dict_not_exception():
    assert gpt_service.repair_json("совсем не json") == {}


# --------------------------------------------------------------- разбор ответа

def test_field_synonyms_are_accepted():
    """Эталон из урока называет поля strong_sides/weak_sides — принимаем оба варианта."""
    a = Assessment.from_payload({"score": 72, "strong_sides": ["MDR"],
                                 "weak_sides": ["нет аудита"]})
    assert a.score == 72 and a.strengths == ["MDR"] and a.weaknesses == ["нет аудита"]


def test_single_string_instead_of_list_is_accepted():
    a = Assessment.from_payload({"score": 50, "strengths": "опыт с MDR"})
    assert a.strengths == ["опыт с MDR"]


def test_score_is_clamped_and_survives_a_string():
    assert Assessment.from_payload({"score": "85"}).score == 85
    assert Assessment.from_payload({"score": 140}).score == 100
    assert Assessment.from_payload({"score": -5}).score == 0
    assert Assessment.from_payload({"score": "не знаю"}).score == 0


# --------------------------------------------------------------- вызов модели

def test_assess_parses_a_good_answer_without_network():
    answer = json.dumps({"score": 78, "strengths": ["MDR"], "weaknesses": [],
                         "missing_skills": ["IVDR"], "summary": "подходит"})
    assessment, error = gpt_service.assess(VACANCY, RESUME_GOOD,
                                           caller=lambda m, model: answer)
    assert error == "" and assessment.score == 78
    assert assessment.missing_skills == ["IVDR"]


def test_assess_retries_and_then_reports_instead_of_raising():
    calls = {"n": 0}

    def flaky(messages, model):
        calls["n"] += 1
        raise RuntimeError("timeout")

    assessment, error = gpt_service.assess(VACANCY, RESUME_GOOD, retries=2, caller=flaky)
    assert calls["n"] == 2
    assert assessment.score == 0 and "timeout" in error


def test_assess_asks_for_json_in_the_prompt():
    """Просить формат заранее дешевле, чем чинить ответ каждый раз."""
    assert "JSON" in gpt_service.SYSTEM_PROMPT
    messages = gpt_service.build_messages(VACANCY, RESUME_GOOD, "Иванов")
    assert messages[0]["role"] == "system"
    assert "Иванов" in messages[1]["content"]
    assert "ВАКАНСИЯ" in messages[1]["content"] and "РЕЗЮМЕ" in messages[1]["content"]


def test_default_model_is_the_cheap_one():
    assert gpt_service.DEFAULT_MODEL == "gpt-4o-mini"


# --------------------------------------------------------------- история

def test_history_survives_a_broken_file(tmp_path):
    """Битая история не должна ронять приложение."""
    broken = tmp_path / "history.json"
    broken.write_text("{не json", encoding="utf-8")
    assert scoring.load_history(broken) == []


def test_result_is_written_and_read_back(tmp_path):
    path = tmp_path / "history.json"
    result = scoring.build_result("Иванов", "RA", VACANCY, RESUME_GOOD,
                                  Assessment(score=70, summary="ок"),
                                  scoring.DEFAULT_WEIGHTS, model="gpt-4o-mini")
    scoring.save_result(result, path)
    items = scoring.load_history(path)
    assert len(items) == 1 and items[0]["candidate"] == "Иванов"
    rows = scoring.history_rows(items)
    assert rows[0]["Итог"] == result.final_score
    assert rows[0]["Уровень"] == result.breakdown.seniority


def test_newest_result_comes_first(tmp_path):
    path = tmp_path / "history.json"
    for name in ("Первый", "Второй"):
        scoring.save_result(
            scoring.build_result(name, "RA", VACANCY, RESUME_GOOD, Assessment(score=50),
                                 scoring.DEFAULT_WEIGHTS), path)
    assert scoring.load_history(path)[0]["candidate"] == "Второй"


def test_nameless_candidate_does_not_break_history(tmp_path):
    result = scoring.build_result("", "RA", VACANCY, RESUME_GOOD, Assessment(score=10),
                                  scoring.DEFAULT_WEIGHTS)
    assert result.candidate == "Без имени"


# --------------------------------------------------------------- данные вакансий

def test_vacancies_file_is_about_our_domain():
    """Задание решается на нашей теме, а не на абстрактных ИТ-вакансиях."""
    data = json.loads((ROOT / "data" / "vacancies.json").read_text(encoding="utf-8"))
    assert len(data) >= 3
    joined = " ".join(v["text"] for v in data).lower()
    for term in ("mdr", "iso 13485", "нотифицированн"):
        assert term in joined
    for vacancy in data:
        assert vacancy["title"] and len(vacancy["text"]) > 200
