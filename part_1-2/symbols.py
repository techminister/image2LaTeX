# https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols


alphabet_upper = [chr(num) for num in range(ord("A"), ord("Z") + 1)]
alphabet_lower = [chr(num) for num in range(ord("a"), ord("z") + 1)]
numerals = [str(num) for num in range(10)]
# Note: Capitals of some characters (Alpha, Beta, Epsilon, ...) use \mathrm or \text because they look like upright
# Roman characters, and some TeX engines do not support \Alpha etc. Hence they are not included under "greek".
greek = [
	r"\alpha", r"\beta", r"\Gamma", r"\gamma", r"\Delta", r"\delta",
	r"\epsilon", r"\varepsilon", r"\zeta", r"\eta", r"\Theta", r"\theta", r"\vartheta",
	r"\iota", r"\kappa", r"\Lambda", r"\lambda", r"\mu",
	r"\nu", r"\Xi", r"\xi", r"\Pi", r"\pi", r"\varpi",
	r"\rho", r"\varrho", r"\Sigma", r"\sigma", r"\varsigma", r"\tau", r"\Upsilon", r"\upsilon",
	r"\Phi", r"\phi", r"\varphi", r"\chi", r"\Psi", r"\psi", r"\Omega", r"\omega"
]
# \mathrm is identical to \text.
mathrm = [r"\mathrm{" + char + r"}" for char in alphabet_lower + alphabet_upper]
# Lowercase \mathcal and \mathbb are identical to alphabet_lower.
mathcal = [r"\mathcal{" + char + r"}" for char in alphabet_upper]
mathbb = [r"\mathbb{" + char + r"}" for char in alphabet_upper]
unary = [r"+", r"-", r"\neg", r"!", r"\#"]
relation = [
	r"<", r"\nless", r"\leq", r"\leqslant", r"\nleq", r"\nleqslant",
	r"\prec", r"\nprec", r"\preceq", r"\npreceq", r"\ll", r"\lll",
	r"\subset", r"\not\subset", r"\subseteq", r"\nsubseteq", r"\sqsubset", r"\sqsubseteq",
	r">", r"\ngtr", r"\geq", r"\geqslant", r"\ngeq", r"\ngeqslant", 
	r"\succ", r"\nsucc", r"\succeq", r"\nsucceq", r"\gg", r"\ggg",
	r"\supset", r"\not\supset", r"\supseteq", r"\nsupseteq", r"\sqsupset", r"\sqsupseteq",
	r"=", r"\doteq", r"\equiv", r"\approx", r"\cong", r"\simeq", r"\sim", r"\propto", r"\neq",
	r"\parallel", r"\asymp", r"\vdash", r"\in", r"\smile", r"\models", r"\perp",
	r"\nparallel", r"\bowtie", r"\dashv", r"\ni", r"\frown", r"\notin", r"\mid"
]
binary = [
	r"\pm", r"\mp", r"\times", r"\div", r"\ast", r"\star", r"\dagger", r"\ddagger",
	r"\cap", r"\cup", r"\uplus", r"\sqcap", r"\sqcup", r"\vee", r"\wedge", r"\cdot",
	r"\diamond", r"\bigtriangleup", r"\bigtriangledown", r"\triangleleft", r"\triangleright", r"\bigcirc", r"\bullet", r"\wr",
	r"\oplus", r"\ominus", r"\otimes", r"\oslash", r"\odot", r"\circ", r"\setminus", r"\amalg"
]
negated_binary = [
	r"\nleqq", r"\lneq", r"\lneqq", r"\lvertneqq", r"\lnsim", r"\lnapprox",
	r"\precneqq", r"\precnsim", r"\precnapprox", r"\nsim", r"\nshortmid", r"\nmid",
	r"\nvdash", r"\nVdash", r"\ntriangleleft", r"\ntrianglelefteq", r"\nsubseteqq", r"\subsetneq", r"\varsubsetneq", r"\subsetneqq", r"\varsubsetneqq",
	r"\ngeqq", r"\gneq", r"\gneqq", r"\gvertneqq", r"\gnsim", r"\gnapprox",
	r"\succneqq", r"\succnsim", r"\succnapprox", r"\ncong", r"\nshortparallel",
	r"\nvDash", r"\nVDash", r"\ntriangleright", r"\ntrianglerighteq", r"\nsupseteqq", r"\supsetneq", r"\varsupsetneq", r"\supsetneqq", r"\varsupsetneqq"
]
set_notation = [
	r"\O", r"\varnothing", r"\exists", r"\nexists", r"\forall", r"\iff", "\leftrightarrow", r"\top", r"\bot"
]
geometry = [
	r"\angle", r"\triangle", r"\measuredangle", r"\square", r"\not\perp"
]
delimiters = [
	r"|", r"(", r"\{", r"\lceil", r"\ulcorner", r"\|", r")", r"\}", r"\rceil", r"\urcorner",
	r"/", r"[", r"\langle", r"\lfloor", r"\llcorner", r"\backslash", r"]", r"\rangle", r"\rfloor", r"\lrcorner"
]
arrows = [
	r"\rightarrow", r"\mapsto", "\leftarrow", r"\Rightarrow", r"\Leftarrow",
	r"\longrightarrow", r"\longmapsto", r"\longleftarrow", r"\Longrightarrow", r"\Longleftarrow",
	r"\uparrow", r"\downarrow", r"\updownarrow", r"\Uparrow", r"\Downarrow", r"\Updownarrow"
]
others = [
	r"\partial", r"\eth", r"\hbar", r"\imath", r"\jmath", r"\ell", r"\Re", r"\Im", r"\wp", r"\nabla", r"\Box", r"\infty"
]


symbols = alphabet_upper + alphabet_lower + numerals + greek + mathrm + mathcal + mathbb \
	+ unary + relation + binary + negated_binary + set_notation + geometry + delimiters + arrows + others
