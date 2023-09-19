%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Masters/Doctoral Thesis 
% Class File
% Version 1.6 (27/8/17)
%
% This class was downloaded from:
% http://www.LaTeXTemplates.com
%
% Authors:
% Vel (vel@latextemplates.com)
% Johannes Böttcher
%
% Notes:
% 1) This class file defines the structure and layout of the template file (main.tex).
% 2) It has been written in such a way that under most circumstances you should not need
% to edit it; updating it to a newer version will be harder. If you do make changes, please change the name of
% the file and add comments to make your changes more visible.
%
% Class license:
% LPPL v1.3c (http://www.latex-project.org/lppl)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----------------------------------------------------------------------------------------
%	CLASS DEFINITION AND PARAMETERS
%----------------------------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}[1996/12/01]
\newcommand{\classname}{MastersDoctoralThesis}
\ProvidesClass{\classname}[2017/08/27 v1.6 LaTeXTemplates.com]
\providecommand{\baseclass}{book}
\RequirePackage{etoolbox}
\RequirePackage{xparse}
\newbool{nolistspace}
\newbool{chapteroneline}
\newbool{listtoc}
\newbool{toctoc}
\newbool{parskip}
\newbool{hyperrefsupport}
\booltrue{hyperrefsupport}
\newbool{headsepline}
\newbool{consistentlayout}


\usepackage{xspace}
\usepackage{here}
\usepackage{dsfont}
\usepackage{amsmath}
\usepackage{physics}
\usepackage{ragged2e}
\usepackage{hyphenat}

\DeclareOption{nohyperref}{\boolfalse{hyperrefsupport}}
\DeclareOption{nolistspacing}{\booltrue{nolistspace}}
\DeclareOption{liststotoc}{\booltrue{listtoc}}
\DeclareOption{chapterinoneline}{\booltrue{chapteroneline}}
\DeclareOption{toctotoc}{\booltrue{toctoc}}
\DeclareOption{parskip}{\booltrue{parskip}}
\DeclareOption{headsepline}{\booltrue{headsepline}}
\DeclareOption{consistentlayout}{\booltrue{consistentlayout}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\baseclass}}

\ProcessOptions\relax

\LoadClass{\baseclass}

%  Simple interface for the user  to customize the chapter titles 
\ProvideDocumentCommand{\abovechapterskip}{}{\vspace*{20pt}}
\ProvideDocumentCommand{\chapterbelowskip}{}{\vspace*{40pt}}
\ProvideDocumentCommand{\chapterinbetweenskip}{}{\vspace*{20pt}}
\ProvideDocumentCommand{\autodot}{}{}
\ProvideDocumentCommand{\mdtChapapp}{}{}
\ProvideDocumentCommand{\chapteralign}{}{\raggedright}
\ProvideDocumentCommand{\chapterfont}{}{\Huge\bfseries}
\ProvideDocumentCommand{\chapterprefixfont}{}{\LARGE\bfseries}
\DeclareDocumentCommand{\@makechapterhead}{ m }{%
	\abovechapterskip
	{\parindent \z@ \chapteralign \normalfont
		\ifnum \c@secnumdepth >\m@ne
		\if@mainmatter
		\ifbool{chapteroneline}{%
			\chapterfont \mdtChapapp\thechapter\autodot\enspace
		}{%
			\chapterprefixfont \@chapapp\space \thechapter
			\par\nobreak
			\chapterinbetweenskip
		}%
		\fi
		\fi
		\interlinepenalty\@M%
		\chapterfont #1\par\nobreak
		\chapterbelowskip
	}
	\thispagestyle{\chapter@p@gestyle}
}
\def\@makeschapterhead#1{%
	\abovechapterskip
	{\parindent \z@ \chapteralign
		\normalfont
		\interlinepenalty\@M
		\chapterfont  #1\par\nobreak
		\chapterbelowskip
	}
	\thispagestyle{\chapter@p@gestyle}
}


% Addchap provides unnumbered chapters with an entry in the table of contents as well as an updated header
\ProvideDocumentCommand{\addchap}{ s o m }{%
	\chapter*{#3}%
	\markboth{}{}%
	\IfBooleanTF{#1}{%
	}{%
		\IfNoValueTF{#2}{%
			\addchaptertocentry{#3}%
			\markboth{\MakeMarkcase{#3}}{\MakeMarkcase{#3}}%
		}{%
			\addchaptertocentry{#2}%
			\markboth{\MakeMarkcase{#2}}{\MakeMarkcase{#2}}%
		}%
	}%
}%

\ProvideDocumentCommand{\addsec}{ s o m }{%
	\section*{#3}%
	\markright{}%
	\IfBooleanTF{#1}{%
	}{%
		\IfNoValueTF{#2}{%
			\addcontentsline{toc}{section}{#3}%
			\markright{\MakeMarkcase{#3}}%%
		}{%
			\addcontentsline{toc}{section}{#2}%
			\markright{\MakeMarkcase{#2}}%
		}%
	}%
}%

%----------------------------------------------------------------------------------------
%	CLASS OPTIONS
%----------------------------------------------------------------------------------------

\ifbool{parskip}{\RequirePackage{parskip}} % If the parskip option is passed to the class, require the parskip package 


\ifbool{listtoc}{% If the liststotoc option has been passed to the class, add the lists to the table of contents
	\patchcmd{\listoftables}{\@starttoc{lot}}{%
		\addchaptertocentry{\listtablename}\@starttoc{lot}%
	}{}{}%
	\patchcmd{\listoffigures}{\@starttoc{lof}}{%
		\addchaptertocentry{\listfigurename}\@starttoc{lof}%
	}{}{}%
}

\ifbool{toctoc}{% If the toctotoc options has been passed to the class, add the table of contents to the table of contents
	\patchcmd{\tableofcontents}{\@starttoc{toc}%
}{%
	\addchaptertocentry{\contentsname}\@starttoc{toc}}{}{}%
}

\patchcmd{\tableofcontents}{\MakeUppercase}{\MakeMarkcase}{}{}
\patchcmd{\tableofcontents}{\MakeUppercase}{\MakeMarkcase}{}{}
\patchcmd{\listoffigures}{\MakeUppercase}{\MakeMarkcase}{}{}
\patchcmd{\listoffigures}{\MakeUppercase}{\MakeMarkcase}{}{}
\patchcmd{\listoftables}{\MakeUppercase}{\MakeMarkcase}{}{}
\patchcmd{\listoftables}{\MakeUppercase}{\MakeMarkcase}{}{}

% If the option `nolistspacing' is given, the spacing in the different lists is reduced to single spacing. This option is only useful, if the spacing of the document has been changed to onehalfspacing or doublespacing.
\ifbool{nolistspace}{
	\patchcmd{\listoffigures}{%
		\@starttoc{lof}
	}{%
		\begingroup%
		\singlespace\@starttoc{lof}\endgroup%
	}{}{}%
	\patchcmd{\listoftables}{%
		\@starttoc{lot}
	}{%
		\begingroup%
		\singlespace\@starttoc{lot}\endgroup%
	}{}{}%
	\patchcmd{\tableofcontents}{%
		\@starttoc{toc}
	}{%
		\begingroup%
		\singlespace\@starttoc{toc}\endgroup%
	}{}{}%
}{}


%----------------------------------------------------------------------------------------
%	REQUIRED PACKAGES
%----------------------------------------------------------------------------------------
\RequirePackage[utf8]{inputenc} 
\RequirePackage[english,french]{babel} % Required for automatically changing names of document elements to languages besides english
\hyphenpenalty=7000
\exhyphenpenalty=7000
\RequirePackage{scrbase} % Required for handling language-dependent names of sections/document elements

\RequirePackage{scrhack} % Loads fixes for various packages

\RequirePackage{setspace} % Required for changing line spacing

\RequirePackage{longtable} % Required for tables that span multiple pages (used in the symbols, abbreviations and physical constants pages)

\RequirePackage{siunitx} % Required for \SI commands

\RequirePackage{graphicx} % Required to include images
\graphicspath{{Figures/}{./}} % Specifies where to look for included images

\RequirePackage{booktabs} % Required for better table rules

\RequirePackage{caption} % Required for customising the captions
\usepackage{subcaption}
\captionsetup{justification=centerlast,font=small,labelfont=sc,margin=50pt}

\RequirePackage{ifpdf}
\RequirePackage[usenames,dvipsnames]{color} % pour les couleurs
\RequirePackage{textcomp}
%\RequirePackage{titlesec}
\RequirePackage{parcolumns}
\RequirePackage{multicol}
\RequirePackage[T1]{fontenc} % permet de sp\'{e}cifier \`{a} LaTeX l'utilisation du codage de caract\`{e}res T1, nouvelle norme LaTeX non utilis\'{e}e par d\'{e}faut pour des raisons de compatibilit\'{e} avec les anciens documents LaTeX.
\RequirePackage{lmodern} % Assurer une bonne impression!
\RequirePackage{tikz} % tikz est utilise pour tracer des boites, par exemple
\usepackage[compat=1.0.0]{tikz-feynman}
\usepackage{multirow}
\RequirePackage{fix-cm}
\RequirePackage{graphicx} % Pour ins\'{e}rer des images. Utiliser le format jpg pour plus de simplicit\'{e}.
%\RequirePackage{sectsty} % obligatoire, pour jouer avec l'apparence des titres
\RequirePackage{shorttoc} % pour le sommaire, au debut.
%\RequirePackage{fancyhdr} % Afin de r\'{e}aliser soi-même les en-têtes et pieds de page, voir chaque d\'{e}but de chapitre.
\RequirePackage{pifont} % Pour utiliser des symboles divers.
%\RequirePackage{color}
\RequirePackage{comment}
\RequirePackage{wallpaper}
%\RequirePackage[inner=30mm,outer=20mm,top=30mm,bottom=20mm]{geometry}
\RequirePackage{setspace}
%Ou bien : \RequirePackage[top=4 cm, bottom=4 cm, left=3 cm, right=3 cm]{geometry} % On peut modifier ces valeurs pour augmenter ou r\'{e}duire les marges. Attention aux en-têtes de chapitres. 
\RequirePackage{epigraph} % \cleartoevenpage
%\RequirePackage{geometry}
\RequirePackage{etoolbox}

\RequirePackage{setspace} % permet d'utiliser les commandes \spacing, doublespace (double interligne), singlespace (simple interligne) et onehalfspace (un interligne et demi)

\RequirePackage{hyperref} %Utiliser pour une bibliographie interactive + sommaire et TOC interactifs (pour une sortie PDF, par exemple).
%\hypersetup{colorlinks=true, citecolor=black, filecolor=black, linkcolor=black, urlcolor=black}

\RequirePackage{eso-pic} % For \AddToShipoutPicture used for cover backgrounds
\RequirePackage{framed} % utilise pour coloriser les encarts


% Define commands to set fonts throughout the document
\newcommand*{\selectfontfrontcover}{\fontfamily{phv}\selectfont}  % Font style used in front cover 
\newcommand*{\selectfontbackcover}{\fontfamily{phv}\selectfont}   % Font style used in back cover 
\newcommand*{\selectfontchapheads}{\fontfamily{phv}\selectfont} % Font style used chapter headings

%----------------------------------------------------------------------------------------
%	DEFINE CUSTOM THESIS INFORMATION COMMANDS
%----------------------------------------------------------------------------------------

%\NewDocumentCommand{\thesistitle} { o m }{%
% \IfValueTF{#1}{\def\shorttitle{#1}}{\def\shorttitle{#2}}%
% \def\@title{#2}%
% \def\ttitle{#2}%
%}
%\DeclareDocumentCommand{\author}{m}{\newcommand{\authorname}{#1}\renewcommand{\@author}{#1}}
%\NewDocumentCommand{\supervisor}{m}{\newcommand{\supname}{#1}}
%\NewDocumentCommand{\examiner}{m}{\newcommand{\examname}{#1}}
%\NewDocumentCommand{\degree}{m}{\newcommand{\degreename}{#1}}
%\NewDocumentCommand{\addresses}{m}{\newcommand{\addressname}{#1}}
%\NewDocumentCommand{\university}{m}{\newcommand{\univname}{#1}}
%\NewDocumentCommand{\department}{m}{\newcommand{\deptname}{#1}
%\NewDocumentCommand{\group}{m}{\newcommand{\groupname}{#1}}
%\NewDocumentCommand{\faculty}{m}{\newcommand{\facname}{#1}}
%\NewDocumentCommand{\subject}{m}{\newcommand{\subjectname}{#1}}
%\NewDocumentCommand{\keywords}{m}{\newcommand{\keywordnames}{#1}}

\newcommand{\checktoopen}{% New command to move content to the next page which prints to the next odd page if twosided mode is active  
	\if@openright\cleardoublepage\else\clearpage\fi
	\ifdef{\phantomsection}{\phantomsection}{}% The \phantomsection command is necessary for hyperref to jump to the correct page
}

\NewDocumentCommand{\bhrule}{}{\typeout{--------------------}}
\NewDocumentCommand{\tttypeout}{m}{\bhrule\typeout{\space #1}\bhrule}

\newcommand{\HRule}{\rule{.9\linewidth}{.6pt}} % New command to make the lines in the title page
\newcommand{\decoRule}{\rule{.8\textwidth}{.4pt}} % New command for a rule to be used under figures

\setcounter{tocdepth}{3} % The depth to which the document sections are printed to the table of contents
\ProvideDocumentCommand{\addchaptertocentry}{ m }{%
	\addcontentsline{toc}{chapter}{#1}%
}

%----------------------------------------------------------------------------------------
%	COLOURS
%----------------------------------------------------------------------------------------

\usepackage{xcolor} % Required for specifying custom colours

\colorlet{mdtRed}{red!50!black}

%----------------------------------------------------------------------------------------
%	MARGINS
%----------------------------------------------------------------------------------------

\RequirePackage{geometry}
\geometry{
	headheight=4ex,
	includehead,
	includefoot
}

\raggedbottom
\usepackage{cleveref}
%----------------------------------------------------------------------------------------
%	PENALTIES
%----------------------------------------------------------------------------------------

%\doublehyphendemerits=10000 % No consecutive line hyphens
%\brokenpenalty=10000 % No broken words across columns/pages
%\widowpenalty=9999 % Almost no widows at bottom of page
%\clubpenalty=9999 % Almost no orphans at top of page
%\interfootnotelinepenalty=9999 % Almost never break footnotes

%----------------------------------------------------------------------------------------
%	HEADERS AND FOOTERS
%----------------------------------------------------------------------------------------

\RequirePackage[markcase=used]{scrlayer-scrpage}
\providepairofpagestyles{thesisSimple}{%
	\clearpairofpagestyles%
	\automark[chapter]{chapter}
	\ihead{\headmark}% Inner header
	\ohead[\pagemark]{\pagemark}% Outer header
}
\ifoot{}% Inner footer
\ofoot{}% Outer footer
\pagestyle{thesisSimple}
\providepairofpagestyles[thesisSimple]{thesis}{%
	\automark*[section]{}%
}
\providepairofpagestyles[thesisSimple]{review}{%
	\ofoot[\shorttitle/\authorname]{\shorttitle/\authorname}
	\ifoot[\today]{\today}
}
\pagestyle{thesis}
\ifbool{headsepline}{\KOMAoption{headsepline}{true}}{}
\PreventPackageFromLoading[\ClassError{\classname}{Package `fancyhdr' is
incompatible\MessageBreak with this class}{The pagesyles are defined 
	using package `scrlayer-scrpage', please consult the\MessageBreak 
KOMA-script documentation for details.}]{fancyhdr}


\newcommand{\blank@p@gestyle}{empty}
\newcommand{\chapter@p@gestyle}{plain}
\NewDocumentCommand{\blankpagestyle}{ m }{%
	\ClassWarning{\classname}{\string\blankpagestyle\space is
	obsolete,\MessageBreak use \string\setblankpagestyle \space  instead}\renewcommand{\blank@p@gestyle}{}{#1}
}
\NewDocumentCommand{\setblankpagestyle}{ m }{\renewcommand{\blank@p@gestyle}{#1}}
\NewDocumentCommand{\setchapterpagestyle}{ m }{\renewcommand{\chapter@p@gestyle}{#1}}

\DeclareDocumentCommand\cleardoublepage{}{\clearpage\if@twoside \ifodd\c@page\else
	\hbox{}
	\thispagestyle{\blank@p@gestyle}
	\newpage
	\if@twocolumn\hbox{}\newpage\fi\fi\fi%
}

%----------------------------------------------------------------------------------------
%	ABBREVIATIONS PAGE DESIGN
%----------------------------------------------------------------------------------------

\newcommand{\abbrevname}{List of Abbreviations}
\providecaptionname{english,british,american}{\abbrevname}{List of Abbreviations}
\providecaptionname{ngerman,german,austrian,naustrian}{\abbrevname}{Abk\"urzungsverzeichnis}
\NewDocumentEnvironment{abbreviations}{ m }{%
	\ifbool{nolistspace}{\begingroup\singlespacing}{}
	\ifbool{listtoc}{\addchap{\abbrevname}}{\addchap*{\abbrevname}}
	\begin{longtable}{#1}
	}{%
	\end{longtable}
	\addtocounter{table}{-1}% Don't count this table as one of the document tables
	\ifbool{nolistspace}{\endgroup}{}
}

%----------------------------------------------------------------------------------------
%	ABSTRACT PAGE DESIGN
%----------------------------------------------------------------------------------------

%\DeclareDocumentCommand{\abstractauthorfont}{}{}
%\DeclareDocumentCommand{\abstracttitlefont}{}{}
%\newcommand{\byname}{by}
%\newcommand{\abstractname}{Abstract}
%\providecaptionname{german,ngerman,austrian,naustrian}{\byname}{von}
%\providecaptionname{american,australian,british,canadian,english,newzealand,UKenglish,USenglish}{\byname}{by}
%\ifbool{consistentlayout}{
%	\DeclareDocumentEnvironment{abstract}{ O{} }{%
%		\addchap*{\abstractname}%
%		{\chapteralign\normalsize\abstractauthorfont \authorname \par}% Author name
%		\vspace{\baselineskip}
%		{\chapteralign\parbox{.7\linewidth}{\chapteralign\normalsize\itshape\abstracttitlefont\@title}\par}% Thesis title
%		\bigskip\noindent\ignorespaces
%	}%
%	{}%end alt-abstract
%}{%
%	\DeclareDocumentEnvironment{abstract}{ O{\null\vfill} }{
%		\checktoopen
%		\tttypeout{\abstractname}
%		#1%added to be able to have abstract more than one page long
%		\thispagestyle{plain}
%		\begin{center}
%			{\normalsize \MakeUppercase{\univname} \par}% University name in capitals
%			\bigskip
%			{\huge\textit{\abstractname} \par}
%			\bigskip
%			{\normalsize \facname \par}% Faculty name
%			{\normalsize \deptname \par}% Department name
%			\bigskip
%			{\normalsize \degreename\par}% Degree name
%			\bigskip
%			{\normalsize\bfseries \@title \par}% Thesis title
%			\medskip
%			{\normalsize \byname{} \authorname \par}% Author name
%			\bigskip
%		\end{center}
%	}
%	{
%		\vfill\null
%	}
%}
%
%\DeclareDocumentEnvironment{extraAbstract}{ O{\null\vfill} }{
%	\checktoopen
%	\tttypeout{\abstractname}
%	#1%added to be able to have abstract more than one page long
%	\thispagestyle{empty}
%	\begin{center}
%		{\normalsize \MakeUppercase{\univname} \par}% University name in capitals
%		\bigskip
%		{\huge\textit{\abstractname} \par}
%		\bigskip
%		{\normalsize \facname \par}% Faculty name
%		{\normalsize \deptname \par}% Department name
%		\bigskip
%		{\normalsize \degreename\par}% Degree name
%		\bigskip
%		{\normalsize\bfseries \@title \par}% Thesis title
%		\medskip
%		{\normalsize \byname{} \authorname \par}% Author name
%		\bigskip
%	\end{center}
%}
%{
%	\vfill\null
%}

%----------------------------------------------------------------------------------------
%	ACKNOWLEDGEMENTS PAGE DESIGN
%----------------------------------------------------------------------------------------

\usepackage{xcolor}
\colorlet{mdtRed}{red!50!black}
\newcommand{\acknowledgementname}{Remerciements}
\providecaptionname{american,australian,british,canadian,english,newzealand,UKenglish,USenglish} {\acknowledgementname}{Remerciements} % Acknowledgement text for English countries
\providecaptionname{german,ngerman,austrian,naustrian}{\acknowledgementname}{Danksagung} % Acknowledgement text for Germanic countries

\ifbool{consistentlayout}{
	\DeclareDocumentEnvironment{acknowledgements}{}{%
		\tttypeout{\acknowledgementname}
		\addchap*{\acknowledgementname}
	}
}
{
	\DeclareDocumentEnvironment{acknowledgements}{}{%
		\checktoopen
		\tttypeout{\acknowledgementname}
		\thispagestyle{plain}
		\begin{center}{\huge\textit{\acknowledgementname}\par}\end{center}
	}
	{
		\vfil\vfil\null
	}
}


%----------------------------------------------------------------------------------------
%	DEDICATION PAGE DESIGN
%----------------------------------------------------------------------------------------

\ifbool{consistentlayout}{
	\DeclareDocumentCommand{\dedicatory}{
		m O{\vspace*{.7\textheight} }  }{
			\checktoopen\tttypeout{Dedicatory}
			\markboth{}{}
			#2
			{\hfill\parbox{.4\textwidth}{\flushright#1\par}}
		}
}{
	\newcommand\dedicatory[1]{
		\checktoopen
		\tttypeout{Dedicatory}
		\null\vfil
		\thispagestyle{plain}
		\begin{center}{\Large\slshape #1}\end{center}
		\vfil\null
	}
}


%----------------------------------------------------------------------------------------
%	PHYSICAL CONSTANTS PAGE DESIGN
%----------------------------------------------------------------------------------------

\newcommand{\constantsname}{Physical Constants}
\providecaptionname{english,british,american}{\constantsname}{Physical Constants}
\providecaptionname{ngerman,german,austrian,naustrian}{\constantsname}{Physikalische Konstanten}

\NewDocumentEnvironment{constants}{ m }{%
	\ifbool{nolistspace}{\begingroup\singlespacing}{}
	\ifbool{listtoc}{\addchap{\constantsname}}{\addchap*{\constantsname}}
	\begin{longtable}{#1}
	}{%
	\end{longtable}
	\addtocounter{table}{-1}% Don't count this table as one of the document tables
	\ifbool{nolistspace}{\endgroup}{}
}

%----------------------------------------------------------------------------------------
%	SYMBOLS PAGE DESIGN
%----------------------------------------------------------------------------------------

\newcommand{\symbolsname}{List of Symbols}
\providecaptionname{english,british,american}{\symbolsname}{List of Symbols}
\providecaptionname{ngerman,german,austrian,naustrian}{\symbolsname}{Symbolverzeichnis}

\NewDocumentEnvironment{symbols}{ m }{%
	\ifbool{nolistspace}{\begingroup\singlespacing}{}
	\ifbool{listtoc}{\addchap{\symbolsname}}{\addchap*{\symbolsname}}
	\begin{longtable}{#1}
	}{%
	\end{longtable}
	\addtocounter{table}{-1}% Don't count this table as one of the document tables
	\ifbool{nolistspace}{\endgroup}{}
}

%----------------------------------------------------------------------------------------

\ifbool{hyperrefsupport}{% If the nohyperref class option has not been specified
\AtEndPreamble{\RequirePackage{hyperref}
\hypersetup{pdfpagemode={UseOutlines},
bookmarksopen=true,
bookmarksopenlevel=0,
hypertexnames=false,
colorlinks=true,% Set to false to disable coloring links
citecolor=magenta,% The color of citations
linkcolor=blue,% The color of references to document elements (sections, figures, etc)
urlcolor=mdtRed,% The color of hyperlinks (URLs)
pdfstartview={FitV},
unicode,
breaklinks=true,
}

\pdfstringdefDisableCommands{% If there is an explicit linebreak in a section heading (or anything printed to the pdf-bookmarks), it is replaced by a space
	\let\\\space%
}
	}
}{%nothing
}

%----------------------------------------------------------------------------------------
%---------------------------------------
%-----CLS ED 3M :'( IMT-A ----------------------
%---------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% VARIABLES PAGE DE GARDE %%%%%%%%%%%%%%%%

%%%%% Dossier contenant les info de l'ecole doctorale
\newcommand*{\direcole}[1]{\gdef\vdirecole{./Couverture-these/ecoles-doctorales/#1}}
\direcole{}

%%%%% Nom ecole, une variable par ligne
\newcommand{\nomecoleA}[1]{\gdef\@nomecoleA{#1}}
\nomecoleA{}
\newcommand{\nomecoleB}[1]{\gdef\@nomecoleB{#1}}
\nomecoleB{}

%%%%% Numero ecole doctorale
\newcommand{\numeroecole}[1]{\gdef\@numeroecole{#1}}
\numeroecole{}

%%%% Etablissement delivrant le diplome, une variable par ligne
\newcommand{\nometablissementA}[1]{\gdef\vnometablissementA{#1}}
\nometablissementA{}
\newcommand{\nometablissementB}[1]{\gdef\vnometablissementB{#1}}
\nometablissementB{}
\newcommand{\nometablissementC}[1]{\gdef\vnometablissementC{#1}}
\nometablissementC{}
\newcommand{\nometablissementD}[1]{\gdef\vnometablissementD{#1}}
\nometablissementD{}
\newcommand{\nometablissementE}[1]{\gdef\vnometablissementE{#1}}
\nometablissementE{}

%%%% Logos etablissement delivrant le diplome, supporte deuble affiliation
\newcommand*{\logoetablissementA}[1]{\gdef\vlogoetablissementA{#1}}
\logoetablissementA{}
\newcommand*{\logoetablissementB}[1]{\gdef\vlogoetablissementB{#1}}
\logoetablissementB{}

%%%% Hauteur des logos, variable selon les (double) affiliations
\newcommand*{\hauteurlogoecole}[1]{\gdef\vhauteurlogoecole{#1}}
\hauteurlogoecole{2.4cm}
\newcommand*{\hauteurlogoetablissementA}[1]{\gdef\vhauteurlogoetablissementA{#1}}
\hauteurlogoetablissementA{}
\newcommand*{\hauteurlogoetablissementB}[1]{\gdef\vhauteurlogoetablissementB{#1}}
\hauteurlogoetablissementB{2.4cm}

%%%% Eventuel sous-titre
\newcommand{\lesoustitre}[1]{\gdef\@lesoustitre{#1}}
\lesoustitre{}

%%%% Discipline
\newcommand{\discipline}[1]{\gdef\@discipline{#1}}
\discipline{}

%%%% Jury
\newcommand{\jury}[1]{\gdef\@jury{#1}}
\jury{}

%%%%% Sp\'{e}cialit\'{e}
\newcommand{\spec}[1]{\gdef\@spec{#1}}
\spec{}

%%% Ville de soutenance
\newcommand{\lieu}[1]{\gdef\@lieu{#1}}
\lieu{}

%%% Unite de recherche: laboratoire / department / unit\'{e}
\newcommand{\uniterecherche}[1]{\gdef\@uniterecherche{#1}}
\uniterecherche{}

%%% Num\'{e}ro de la th\`{e}se
\newcommand{\numthese}[1]{\gdef\@numthese{#1}}
\numthese{}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% PAGE DE GARDE %%%%%%%%%%%%%%%%

% Define some font sizes specific to the covers, supposed to be in 12pt
\newcommand{\HugeTwelve}{\fontsize{26}{31}\selectfont} % 12pt \Huge
\newcommand{\LARGETwelve}{\fontsize{20.74}{25}\selectfont} % 12pt \LARGE
\newcommand{\LargeTwelve}{\fontsize{16}{19}\selectfont} % 12pt \Large
\newcommand{\largeTwelve}{\fontsize{14.4}{17}\selectfont} % 12pt \large
\newcommand{\normalTwelve}{\fontsize{12}{13.2}\selectfont} % 12pt \normalsize
\newcommand{\smallTwelve}{\fontsize{11}{13.5}\selectfont} % 12pt \small
\newcommand{\footnotesizeTwelve}{\fontsize{9.5}{11}\selectfont} % 12pt \footnotesize


% Affiche les logos sur les pages de couverture
\newcommand{\displayLogos}{%
  \thispagestyle{empty}
  \begin{tikzpicture}[remember picture,overlay,line width=0mm]
    \node[xshift=6.2cm,yshift=2cm] {
    \parbox{\textwidth}{
      $\vcenter{\hbox{%
        \includegraphics[keepaspectratio,height=\vhauteurlogoecole,%width=7cm
        ]{\vdirecole/logo}%
      }}$
      \hfill
      {\if\vlogoetablissementA\empty \else
      $\vcenter{\hbox{%
        \includegraphics[keepaspectratio,height=\vhauteurlogoetablissementA,width=7cm
        ]{./Couverture-these/logos-etablissements/\vlogoetablissementA}%
      }}$
      \fi}
      \hspace{3mm}
      $\vcenter{\hbox{%
        \includegraphics[keepaspectratio,height=\vhauteurlogoetablissementB,width=7cm
        ]{./Couverture-these/logos-etablissements/\vlogoetablissementB}%
      }}$
      }
  };
  \end{tikzpicture}
  \par\nobreak
}

%mise en page de la page de garde
\makeatletter
\def\maketitle{%
  \thispagestyle{empty}
  \clearpage
  %background image of the front cover
  \AddToShipoutPicture*{%
    \put(0,0){%
    \parbox[b][42.6cm]{\paperwidth}{%
        \vfill
        \includegraphics[width=\paperwidth,keepaspectratio,trim={0 5pt 0 0}]{\vdirecole/image-fond-garde} % Must trim white border off of bottom
        \begin{tikzpicture}
            \fill[color-ecole] (0,0) rectangle (\paperwidth,4.4);
        \end{tikzpicture}
        \vfill
  }}}
  \displayLogos
  %
  \begin{tikzpicture}[remember picture,overlay,line width=0mm]
   \node at (current page.center)
{\parbox{17.6cm}{
\vspace{3.6cm}

\selectfontfrontcover % Set font style for front cover page

{\HugeTwelve \textsc{Th\`{e}se de doctorat de} \\}
% \vspace{5mm}
{\normalTwelve \if\@nomecoleB\empty ~\\ \else \fi} % To compensate the 2 lines of MathSTIC
{\setlength{\baselineskip}{0.9\baselineskip}
{\largeTwelve \if\vnometablissementA\empty ~ \else \textsc{\vnometablissementA} \fi} \\
{\largeTwelve \if\vnometablissementB\empty ~ \else \textsc{\vnometablissementB} \fi} \\
{\largeTwelve \if\vnometablissementC\empty ~ \else \textsc{\vnometablissementC} \fi} \\
{\largeTwelve \if\vnometablissementD\empty ~ \else \textsc{\vnometablissementD} \fi} \\
{\largeTwelve \textsc{\vnometablissementE}} \\
\par}
\vspace{0.01cm}
{\setlength{\baselineskip}{0.7\baselineskip}
{\smallTwelve \textsc{\'{E}cole Doctorale \No \@numeroecole} \vspace{0.1cm}} \\
{\normalTwelve \textit{\@nomecoleA} \vspace{0.1cm}} \\
{\normalTwelve \if\@nomecoleB\empty \else \textit{\@nomecoleB} \\ \fi}
{\normalTwelve Sp\'{e}cialit\'{e} : \textit{\@spec}}

\fontsize{12}{10}\selectfont
\vspace{0.5cm}
\hspace{0.6cm}{\normalTwelve Par \vspace{0.15cm}}
\par}
\hspace{0.6cm}{\LARGETwelve \textbf{\@author}} \vspace{0.5cm}

{\LargeTwelve \textbf{\@title}} \vspace{0.5cm}
	
{\largeTwelve \@lesoustitre} \vspace{0.5cm}
\begin{spacing}{1}
	 \smallTwelve
	 \textbf{Th\`{e}se pr\'{e}sent\'{e}e et soutenue \`{a} \@lieu, le \@date} \\
	 \textbf{Unit\'{e} de recherche : \@uniterecherche} \\
	 \textbf{\if\@numthese\empty \else Th\`{e}se \No : \@numthese \fi} % Hide line if no number provided
\end{spacing}
\vspace{1.3cm}
	\begin{small}
	\begin{spacing}{1}
		 \@jury
	\end{spacing}
	\end{small}
}
};
\end{tikzpicture}
}

\makeatother



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% QUATRIEME DE COUVERTURE %%%%%%%%%%%%%%%%

\newcommand{\backcoverheader}{%
\thispagestyle{empty}
\AddToShipoutPicture*{%
    \put(0,0){%
    \parbox[t][\paperheight]{\paperwidth}{%
        \vspace{-29.6cm}
        \includegraphics[width=\paperwidth,height=\paperheight,keepaspectratio]{\vdirecole/image-fond-dos}%
    }}
    \put(0,0){%
    \parbox[t][\paperheight]{\paperwidth}{%
        \vspace{-14.5cm}
        \includegraphics[width=\paperwidth,height=\paperheight,keepaspectratio]{\vdirecole/image-fond-dos2}%
    }}
}
\hspace{9mm}
\displayLogos
}

\newcommand{\titleFR}[1]{%
\vspace{1cm}
{\centering \noindent \textcolor{color-ecole}{\rule{\textwidth}{0.2cm}}}
\vspace{-1cm}
\selectlanguage{english}
\section*{\selectfontbackcover\smallTwelve \textcolor{color-ecole}{Titre : }{\selectfontbackcover\mdseries{#1}}} % In this particular case, font style needs to get re-selected locally
}

\newcommand{\keywordsFR}[1]{%
\vspace{-0.2cm}
\noindent{\smallTwelve \textbf{Mot cl\'{e}s : }#1}
}

\newcommand{\abstractFR}[1]{%
\vspace{-0.2cm}
\begin{multicols}{2}
\begin{spacing}{1}
	\noindent\smallTwelve \textbf{R\'{e}sum\'{e} : }#1
\end{spacing}
\end{multicols}
}

\newcommand{\titleEN}[1]{%
\vspace{0.5cm}
{\centering \noindent \textcolor{color-ecole}{\rule{\textwidth}{0.2cm}}}
\vspace{-1cm}
\selectlanguage{english}
\section*{\selectfontbackcover\smallTwelve \textcolor{color-ecole}{Title: }{\selectfontbackcover\mdseries{#1}}} % In this particular case, font style needs to get re-selected locally
}

\newcommand{\keywordsEN}[1]{%
\vspace{-0.2cm}
\noindent{\smallTwelve \textbf{Keywords: }#1}
}

\newcommand{\abstractEN}[1]{%
\vspace{-0.2cm}
\begin{multicols}{2}
\begin{spacing}{1}
	\noindent\smallTwelve \textbf{Abstract: }#1
\end{spacing}
\end{multicols}
}


\endinput
% lazyLizardTracer
