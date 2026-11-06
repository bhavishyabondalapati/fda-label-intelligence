from src.extract.scoring import (class_key, f1_from_counts, field_items, gold_fields, name_key, normalize,
                                 pred_fields, score_label, strict_correct)

LABEL = {"generic_name": "SILDENAFIL CITRATE", "substance_name": ["SILDENAFIL CITRATE"], "route": "ORAL",
         "pharm_class_epc": ["Phosphodiesterase 5 Inhibitor [EPC]"], "sections": ["indications_and_usage"]}


def test_normalize():
    assert normalize("  RESPIRATORY (INHALATION) ") == "respiratory inhalation"


def test_salt_forms_match():
    assert name_key("SILDENAFIL CITRATE") == name_key("sildenafil")
    assert name_key("Metoprolol Succinate") == name_key("metoprolol")
    assert name_key("Doxycycline hyclate") == name_key("DOXYCYCLINE")


def test_name_with_only_salt_words_is_kept():
    assert name_key("Sodium Chloride") == frozenset({"sodium", "chloride"})
    assert name_key("Sodium Chloride") != name_key("Sodium")


def test_different_drugs_do_not_match():
    assert name_key("amlodipine") != name_key("atorvastatin")
    assert name_key("amlodipine and benazepril") != name_key("amlodipine")


def test_combination_order_and_and():
    assert name_key("HYDROCODONE BITARTRATE AND ACETAMINOPHEN") == name_key("acetaminophen, hydrocodone")


def test_class_suffix_stripped_case_insensitive():
    assert class_key("Nonsteroidal Anti-inflammatory Drug [EPC]") == class_key("nonsteroidal anti-inflammatory drug")


def test_gold_fields_single_and_combo():
    g = gold_fields(LABEL)
    assert g["is_combination"] is False and g["route"] == "oral" and g["has_boxed_warning"] is False
    combo = {**LABEL, "substance_name": ["HYDROCODONE BITARTRATE", "ACETAMINOPHEN"]}
    assert gold_fields(combo)["is_combination"] is True
    # base + salt of the SAME drug is not a combination
    same = {**LABEL, "substance_name": ["LEVOTHYROXINE SODIUM", "LEVOTHYROXINE"]}
    assert gold_fields(same)["is_combination"] is False


def test_empty_route_is_null_and_empty_class_is_none():
    lab = {**LABEL, "route": "", "pharm_class_epc": []}
    g = gold_fields(lab)
    assert g["route"] is None and g["pharm_class"] == set()
    assert pred_fields({"route": None, "pharm_class": None})["route"] is None
    assert pred_fields({"pharm_class": None})["pharm_class"] == set()


def test_boxed_flag_from_sections():
    assert gold_fields({**LABEL, "sections": ["boxed_warning", "description"]})["has_boxed_warning"] is True


def test_score_label_perfect_and_wrong_route():
    good = {"generic_name": "sildenafil", "active_ingredients": ["Sildenafil"], "is_combination": False,
            "route": "ORAL", "has_boxed_warning": False, "pharm_class": ["phosphodiesterase 5 inhibitor"]}
    assert all(v["correct"] for v in score_label(good, LABEL).values())
    bad = score_label({**good, "route": "INTRAVENOUS"}, LABEL)
    assert bad["route"]["correct"] is False and bad["route"]["items"] == (0, 1, 1)


def test_f1_items_scalar_and_list_rules():
    assert field_items("route", None, "oral") == (0, 0, 1)
    assert field_items("route", "oral", None) == (0, 1, 0)
    assert field_items("route", None, None) == (0, 0, 0)
    assert field_items("pharm_class", {"a", "b"}, {"b", "c"}) == (1, 1, 1)


def test_f1_formula():
    r = f1_from_counts(8, 2, 2)
    assert abs(r["f1"] - 0.8) < 1e-9


def test_strict_names_do_not_strip_salts():
    assert strict_correct("generic_name", "sildenafil", LABEL) is False
    assert strict_correct("generic_name", "Sildenafil citrate", LABEL) is True


def test_form_words_ignored_on_both_sides():
    from src.extract.scoring import FORM_WORDS
    assert name_key("hydrocortisone cream") == name_key("HYDROCORTISONE")
    assert name_key("METOPROLOL SUCCINATE ER TABLETS") == name_key("metoprolol succinate")
    assert name_key("tobramycin ophthalmic solution") == name_key("Tobramycin")
    assert name_key("estradiol") != name_key("estriol cream")
    assert "oral" in FORM_WORDS and len(FORM_WORDS) == 18
    # a name made only of ignored words is kept whole
    assert name_key("Sterile Water for Injection, USP") == name_key("sterile water")
