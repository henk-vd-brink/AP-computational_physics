import functions
import plots

beads, lengths, chains = functions.data(storechains = True, flipping = True)
plots.lengthplot(chains)
plots.beadhist(beads)
plots.chainplot(beads, chains, interval = [0, 1, 2])
