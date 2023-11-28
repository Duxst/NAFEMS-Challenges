# NAFEMS-Challenges
To enhance the use of stochastic methodologies in engineering simulations, the NAFEMS Stochastic Working Group (SWG) periodically poses challenges to the data science and engineering community. The SWG evaluates the solutions to discern the tactics employed by professionals and researchers.

Uncertainty Quantification (UQ) examines the consequences of uncertain data on the creation and optimization of products, aiding in informed decision-making.

There's been a shift in recent years from deterministic design practices, which rely on reserve margins and "worst case" scenarios, to stochastic analysis. This transition seeks to optimize designs or more accurately determine design reliability. Traditional deterministic design can be ambiguous in its provision for uncertainties, leading to over or under-designed solutions. UQ techniques directly tackle these concerns.

UQ is instrumental in comparing simulations to physical tests by factoring in uncertainties in both. This allows for a more accurate comparison than merely juxtaposing two data sets, ensuring dependable decisions. UQ is also utilized during model creation to gauge the impact of model assumptions on predicted outcomes, directing model enhancement or the need for more data.

Here's a breakdown of deterministic vs. probabilistic analysis:

Deterministic Analysis: Uses fixed input values. Often, these are chosen to ensure designs are conservative, leaning towards worst-case scenarios.

Probabilistic Analysis: Considers variability in load and resistance factors, yielding a probability of failure. This method is often termed structural reliability analysis.

Confidence levels indicate the likelihood of reaching a specified percentile. For instance, a 95/90 criterion denotes a 95% probability at the 90th percentile.

Probabilistic approaches differentiate between aleatory (chance-driven) and epistemic (knowledge-limited) uncertainties. The former is random, while the latter stems from information deficits. This differentiation is crucial when interpreting operational data.

Simulation tools, paired with physical tests, are crucial for stochastic assessments. Stochastic analysis entails crafting statistical models to depict uncertain or unknown data. The subsequent steps detail the UQ process:

1. Pinpoint critical simulation input factors.
2. Amass and evaluate data on uncertainties.
3. Categorize data into aleatory or epistemic uncertainties.
4. Conduct sensitivity analysis to validate assumptions.
5. Project the input variability to output metrics.
6. Contrast output variations with set limits for informed design decisions.
7. The Latin Hypercube Sampling (LHC) method boosts the efficiency of the Monte Carlo analysis by considering a comprehensive mix of input parameters.

Monte Carlo analysis calculates the failure probability of a specific mode by running numerous deterministic simulations. These simulations sample load and resistance from distributions. After comparing structural responses to load and resistance for each run, the failure probability is updated. A major constraint of this method is the computational time required to ensure convergence of the failure probability.

These are my attempted solutions to the NAFEMS challenges. The actual challenges can be found at https://www.nafems.org/community/working-groups/stochastics/challenge_problem/
