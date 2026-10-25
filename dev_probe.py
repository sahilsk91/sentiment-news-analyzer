"""dev_probe.py -- Regression checks for the slop/real meter. Run directly."""
from analyzer import analyze

REAL = """According to Union Transport Minister Nitin Gadkari, who said on Monday
that the ministry filed a statement, the data leak affected 42 percent of records.
The CERT-In audit began on October 23, officials said. A court filing documented
11 crore vehicle records. "We are treating this as a serious breach," the
spokesperson said. Reuters reported that the ministry announced a forensic audit
yesterday."""

SLOP = """In today's fast-paced world, Experts Say the digital landscape is
ever-evolving. Moreover, it is important to note that this breakthrough is a
game-changer. Furthermore, the myriad benefits underscore the importance of this
pivotal development. Reportedly, sources say this will revolutionize everything
as we navigate the unprecedented times. In conclusion, only time will tell."""

for name, text in (("REAL", REAL), ("SLOP", SLOP)):
    res = analyze(text)
    s = res["slop"]
    print(f"{name:5s} slop={s['slop_score']:5.1f} real={s['real_score']:5.1f} "
          f"signals={s['signals']}")
    for ev in s["evidence"][:2]:
        print("   ev:", ev[:90])
