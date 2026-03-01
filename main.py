"""
Political Conviction Signal Engine (PCSE) — Main Orchestrator.

Ties together all pipeline layers:
  Layer 1: Data Ingestion (Congress API)
  Layer 2: Baseline Construction (Psycholinguistic profiles)
  Layer 3: Anomaly Detection (Gating model)
  Layer 4: Signal Report

Usage:
  python main.py --mode demo        # Run demo with sample data
  python main.py --mode baseline    # Build baselines from API data
  python main.py --mode detect      # Run detection on new speeches
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.psycholinguistic import PsycholinguisticAnalyzer
from ingestion.congress_api import load_sample_data, SpeechRecord
from baseline.profile_builder import ProfileBuilder, PoliticianProfile
from detection.anomaly_detector import AnomalyDetector


def run_demo():
    """
    Full demo pipeline using sample data.
    Demonstrates: ingestion → analysis → baseline → anomaly detection.
    """
    print("\n" + "🏛️ " * 20)
    print("  POLITICAL CONVICTION SIGNAL ENGINE")
    print("  Demo Mode — Using Sample Data")
    print("🏛️ " * 20)
    
    # =========================================================================
    # LAYER 1: Ingest sample speeches
    # =========================================================================
    print("\n\n📥 LAYER 1: DATA INGESTION")
    print("─" * 50)
    speeches = load_sample_data()
    print(f"Loaded {len(speeches)} sample speeches:")
    for s in speeches:
        print(f"  • {s.speaker_name} ({s.party}) — {s.word_count} words — {s.date}")
    
    # =========================================================================
    # STEP 1.5: Show raw analysis for one speech
    # =========================================================================
    print("\n\n🔬 RAW PSYCHOLINGUISTIC ANALYSIS (single speech demo)")
    print("─" * 50)
    analyzer = PsycholinguisticAnalyzer()
    
    for speech in speeches[:2]:
        features = analyzer.analyze(speech.text)
        if features:
            print(f"\n  {speech.speaker_name} — {speech.context}")
            print(f"  {'.' * 45}")
            print(f"  Authenticity:       {features['authenticity_score']:>6.1f}")
            print(f"  Clout:              {features['clout_score']:>6.1f}")
            print(f"  Cognitive Complex:  {features['cognitive_complexity']:>6.1f}")
            print(f"  Emotional Tone:     {features['emotional_tone']:>6.1f}")
            print(f"  Certainty/Hedging:  {features['certainty_hedging_ratio']:>6.2f}")
            print(f"  I-words:            {features['i_words']:>6.2f}%")
            print(f"  We-words:           {features['we_words']:>6.2f}%")
            print(f"  Certainty:          {features['certainty']:>6.2f}%")
            print(f"  Hedging:            {features['hedging']:>6.2f}%")
    
    # =========================================================================
    # LAYER 2: Build baselines
    # =========================================================================
    print("\n\n📊 LAYER 2: BASELINE CONSTRUCTION")
    print("─" * 50)
    
    # For demo, we need more speeches per politician to build baselines
    # Let's generate synthetic variations to demonstrate the pipeline
    expanded_speeches = _generate_synthetic_history(speeches)
    
    builder = ProfileBuilder()
    profiles = builder.build_profiles(expanded_speeches)
    
    # Print profile summaries
    for pid, profile in profiles.items():
        builder.print_profile_summary(pid)
    
    # =========================================================================
    # LAYER 3: Anomaly Detection
    # =========================================================================
    print("\n\n⚡ LAYER 3: ANOMALY DETECTION")
    print("─" * 50)
    print("Analyzing new speeches against baselines...\n")
    
    detector = AnomalyDetector(profiles)
    
    # Create some "new" speeches that deviate from baseline
    anomalous_speeches = _generate_anomalous_speeches()
    
    signals = detector.analyze_batch(anomalous_speeches)
    
    # Print signal report
    report = detector.get_signal_report(signals)
    print(report)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n\n📋 PIPELINE SUMMARY")
    print("═" * 50)
    print(f"  Speeches ingested:     {len(expanded_speeches)}")
    print(f"  Profiles built:        {len(profiles)}")
    print(f"  Valid baselines:       {sum(1 for p in profiles.values() if p.is_valid)}")
    print(f"  New speeches analyzed: {len(anomalous_speeches)}")
    print(f"  Signals detected:      {len(signals)}")
    
    strong_signals = [s for s in signals if s.strength > 0.6]
    if strong_signals:
        print(f"\n  🔴 STRONG SIGNALS ({len(strong_signals)}):")
        for s in strong_signals:
            print(f"     • {s.politician_name}: {s.signal_type} "
                  f"(strength={s.strength:.2f}, domains={s.policy_domains})")
    
    print(f"\n{'═' * 50}")
    print("  Pipeline complete. In production, signals would be")
    print("  mapped to Kalshi/Polymarket contracts for trading.")
    print(f"{'═' * 50}\n")


def _generate_synthetic_history(base_speeches: list) -> list:
    """
    Generate synthetic speech history for demo purposes.
    In production, this comes from the Congress.gov API.
    
    Creates 40+ speeches per politician with natural variation
    around their base style to establish a meaningful baseline.
    """
    import random
    
    # Templates for generating varied speeches in each politician's style
    templates = {
        "P000197": {  # Pelosi-style: high clout, we-focused, positive
            "phrases": [
                "Mr. Speaker, we must work together to deliver results for the American people. "
                "Our caucus remains united in our commitment to {topic}. "
                "We will continue to fight for working families across this nation. "
                "This is not about politics, this is about people. We have heard from families "
                "in every corner of this country who are counting on us to act. We will not let "
                "them down. Our plan is clear, our resolve is strong, and we will deliver. "
                "The American people elected us to lead, and lead we will. Together we are "
                "stronger than any challenge we face. We must seize this moment and build "
                "a better future for our children and grandchildren. I urge my colleagues to "
                "join us in this effort. We owe it to every family who sent us here.",
                
                "The House has passed landmark legislation on {topic} and we are proud of "
                "the progress we have made. We will not stop until every American has the "
                "opportunity to succeed. Our democratic institutions demand that we act with "
                "urgency and purpose. We have a mandate from the people to deliver change. "
                "This bill represents our values as a nation. We believe in opportunity for "
                "all, not just the privileged few. We will fight for affordable solutions that "
                "work for real families. The stakes are too high for inaction. We must be bold, "
                "we must be decisive, and we must deliver results. That is what leadership "
                "looks like. That is what the American people deserve.",
            ]
        },
        "M001153": {  # Murkowski-style: hedging, deliberative, moderate
            "phrases": [
                "Madam President, I think we should consider the implications of this {topic} "
                "proposal more carefully. Perhaps there are aspects we haven't fully examined. "
                "I believe we might find a better path forward if we take more time to understand "
                "the consequences. It seems like there could be unintended effects on our rural "
                "communities that maybe deserve more attention. Some of my colleagues have raised "
                "concerns that I think are probably worth considering. I'm not entirely sure we "
                "have all the information we need. Maybe we should hold additional hearings and "
                "perhaps invite experts to testify on these questions. I believe a more cautious "
                "approach might serve us better in the long run. It appears that reasonable people "
                "can disagree on this matter, and I think we should respect those differences.",
                
                "I believe there may be room for compromise on {topic} if we approach this with "
                "open minds. Some of my colleagues have raised valid points that perhaps deserve "
                "more consideration. I think we owe it to our constituents to get this right rather "
                "than rush through something that might not work. It seems like the evidence is "
                "somewhat mixed on this approach. Perhaps we should look at what other states have "
                "done and consider whether their experiences might inform our decisions. I'm somewhat "
                "concerned that we may be moving too quickly. Maybe a pilot program would help us "
                "understand the likely impacts before we commit to a full implementation. I think "
                "there are probably better options available if we take the time to explore them.",
            ]
        },
        "C001098": {  # Cruz-style: high certainty, aggressive, fiscal
            "phrases": [
                "Mr. President, this {topic} bill is absolutely reckless and irresponsible. "
                "The American people will never accept this kind of wasteful spending. "
                "I will fight this with every tool at my disposal. We must stop this madness. "
                "Every dollar the government spends is a dollar taken from hardworking taxpayers. "
                "This is outrageous. The Constitution does not give Washington this kind of "
                "unchecked power over our lives. I will absolutely vote no on this legislation "
                "and I will urge every patriotic American to stand against it. We need to cut "
                "spending, reduce the deficit, and restore fiscal sanity. The future of our "
                "republic depends on it. I will never back down from this fight. The American "
                "people deserve better than this reckless, wasteful bill that will crush our economy.",
                
                "I will not back down from opposing this {topic} legislation. This is exactly "
                "the kind of government overreach that destroys trust in our institutions. "
                "We need to fundamentally change how Washington operates. Every family in America "
                "has to balance their budget, but apparently Congress thinks the rules don't apply "
                "to us. This is wrong. This is dangerous. And I will fight it with everything "
                "I have. The American taxpayer is tired of being treated like an ATM machine for "
                "Washington bureaucrats. We must demand accountability. We must demand fiscal "
                "responsibility. I will absolutely stand firm against any attempt to expand "
                "government power at the expense of individual liberty and freedom.",
            ]
        },
        "W000187": {  # Warren-style: analytical, data-driven, nuanced
            "phrases": [
                "Thank you, Mr. Chairman. The data on {topic} is quite revealing. Studies from "
                "multiple institutions show that similar policies have produced mixed results in "
                "other contexts. We need to examine both the costs and benefits carefully before "
                "moving forward with implementation. On one hand, there are potential economic "
                "gains from this approach. On the other hand, we risk creating negative consequences "
                "for working families who are already struggling. The Congressional Budget Office "
                "analysis indicates that the net impact may be smaller than proponents suggest. "
                "I believe we should consider amendments that would strengthen consumer protections "
                "while preserving the core objectives of the legislation. Perhaps a more targeted "
                "approach would yield better outcomes for the people who need help the most.",
                
                "Let me walk through the evidence on {topic} systematically. Research from the "
                "Brookings Institution and the Economic Policy Institute suggests that this kind "
                "of policy typically produces heterogeneous effects across different communities. "
                "Urban areas may benefit while rural communities could face challenges. I think "
                "we need to acknowledge this complexity rather than pretending the answer is "
                "simple. The economic models indicate several possible outcomes depending on "
                "implementation details. I believe we should probably commission additional "
                "analysis before proceeding. However, I also recognize that delay has costs. "
                "Perhaps the best approach is to include sunset provisions and mandatory review "
                "periods so we can evaluate the actual impacts and adjust accordingly.",
            ]
        },
    }
    
    topics = [
        "healthcare", "education", "infrastructure", "climate",
        "immigration", "defense spending", "tax reform", "housing",
        "technology regulation", "trade policy", "social security",
        "criminal justice reform",
    ]
    
    synthetic_speeches = []
    dates_base = ["2024-01-{:02d}", "2024-02-{:02d}", "2024-03-{:02d}",
                  "2024-04-{:02d}", "2024-05-{:02d}"]
    
    for speech in base_speeches:
        sid = speech.speaker_id
        if sid not in templates:
            continue
        
        # Add the original
        synthetic_speeches.append(speech)
        
        # Generate 35-45 more speeches with natural variation
        num_synthetic = random.randint(35, 45)
        for i in range(num_synthetic):
            template = random.choice(templates[sid]["phrases"])
            topic = random.choice(topics)
            text = template.format(topic=topic)
            
            # Add some random variation (extra sentences)
            extras = [
                "This is about our fundamental values as a nation and what we stand for.",
                "We owe this to the next generation of Americans who are counting on us.",
                "History will judge us by our actions today, not our words.",
                "The stakes could not be higher for families across America.",
                "I yield back the balance of my time to the gentleman from Ohio.",
                "I ask unanimous consent to revise and extend my remarks for the record.",
                "The evidence speaks for itself and the data is overwhelming.",
                "Let me be absolutely clear about where I stand on this issue.",
                "This should not be a partisan issue and I hope we can find common ground.",
                "Our constituents are watching and they expect us to deliver real results.",
                "The American dream depends on the decisions we make in this chamber today.",
            ]
            text += " " + " ".join(random.sample(extras, random.randint(3, 5)))
            
            date_template = random.choice(dates_base)
            day = random.randint(1, 28)
            date = date_template.format(day)
            
            synthetic_speeches.append(SpeechRecord(
                text=text,
                speaker_id=sid,
                speaker_name=speech.speaker_name,
                date=date,
                source="synthetic_demo",
                chamber=speech.chamber,
                party=speech.party,
                state=speech.state,
                context=f"Speech on {topic}",
            ))
    
    return synthetic_speeches


def _generate_anomalous_speeches() -> list:
    """
    Generate speeches that deviate from the established baselines.
    These represent the "new data" that the detector analyzes.
    """
    return [
        # Pelosi suddenly hedging (usually high conviction)
        SpeechRecord(
            text="""I think we might need to reconsider our position on this 
            healthcare legislation. It seems like there could be some issues 
            that perhaps we haven't fully addressed. I'm not entirely sure 
            this is the right approach anymore. Maybe we should take a step 
            back and consider alternative strategies. Some members of my caucus 
            have expressed concerns that perhaps deserve more attention. I 
            believe we should probably slow down and really think about the 
            implications. It appears the situation may be more complex than 
            we initially thought. I would suggest we consider forming a 
            working group to study this further before committing to a vote. 
            I'm hopeful we can find a way forward, but I think we need more 
            time to get this right.""",
            speaker_id="P000197",
            speaker_name="Nancy Pelosi",
            date="2024-06-15",
            source="test_anomaly",
            chamber="house",
            party="Democratic",
            state="CA",
            context="Healthcare Bill Discussion — ANOMALOUS",
        ),
        
        # Cruz suddenly moderate and analytical (usually aggressive certainty)
        SpeechRecord(
            text="""I want to acknowledge that this immigration proposal has 
            some merit. Perhaps we should consider the economic data that 
            suggests immigration reform could benefit certain sectors. I think 
            there might be room for a bipartisan approach here. On one hand, 
            we need to maintain border security. On the other hand, our 
            agricultural and technology sectors depend on immigrant labor. 
            I believe we could perhaps find common ground if we focus on 
            the evidence rather than rhetoric. Some of my colleagues across 
            the aisle have made reasonable points that I think deserve 
            consideration. Maybe a phased approach would work best for 
            everyone involved. I'm open to discussions about finding a 
            compromise that serves the national interest.""",
            speaker_id="C001098",
            speaker_name="Ted Cruz",
            date="2024-06-15",
            source="test_anomaly",
            chamber="senate",
            party="Republican",
            state="TX",
            context="Immigration Reform — ANOMALOUS",
        ),
        
        # Warren suddenly emotional and aggressive (usually analytical)
        SpeechRecord(
            text="""I am furious about what Wall Street banks have done to 
            American families. This is outrageous. They destroyed lives, they 
            crushed dreams, and they got away with it. Every single executive 
            who committed fraud must be held accountable. We will absolutely 
            pass this regulation. We must force these corrupt institutions to 
            pay for their crimes. I will not rest, I will not back down, and 
            I will fight until every last predatory lender faces justice. The 
            American people demand action and we will deliver. This is not 
            complicated — they stole from working families and they must face 
            consequences. No more excuses, no more delays. We act now or we 
            betray every family they victimized.""",
            speaker_id="W000187",
            speaker_name="Elizabeth Warren",
            date="2024-06-16",
            source="test_anomaly",
            chamber="senate",
            party="Democratic",
            state="MA",
            context="Financial Regulation — ANOMALOUS",
        ),
    ]


def run_analyzer_demo():
    """Just run the psycholinguistic analyzer demo."""
    from analysis.psycholinguistic import demo
    demo()


def main():
    parser = argparse.ArgumentParser(
        description="Political Conviction Signal Engine (PCSE)"
    )
    parser.add_argument(
        "--mode", 
        choices=["demo", "analyze", "baseline", "detect"],
        default="demo",
        help="Pipeline mode to run"
    )
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        run_demo()
    elif args.mode == "analyze":
        run_analyzer_demo()
    else:
        print(f"Mode '{args.mode}' requires API access. Run with --mode demo first.")
        print("Set CONGRESS_API_KEY environment variable for full functionality.")


if __name__ == "__main__":
    main()
