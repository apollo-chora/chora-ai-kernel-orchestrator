package boot

// Dispatch identity for the ADR-253/254 Pub/Sub lanes. Three strings are in
// play and they are deliberately different, exactly as on the OE lane:
//
//	CrewKind      companion_diagnoser         the ADK agent name; the event AUTHOR
//	DispatchRole  companion_diagnose          what the kennel keys topics on
//	ServiceName   chora-companion-diagnoser   the Deployment; names the subscription
//
// The request subscription for companion_diagnose is one of the two 63-byte
// overrides (chora-companion-diagnoser.agent-dispatch-diagnose-requested), so
// the Deployment sets AGENT_DISPATCH_SUBSCRIPTION explicitly; the derived name
// is only the fallback.

const (
	DispatchRoleDiagnose = "companion_diagnose"
	DispatchRoleExtract  = "companion_extract"
)

// DispatchRole is the dispatch role for a crew kind, or "" when unknown. Empty
// is refused by the boot rather than guessed.
func DispatchRole(crewKind string) string {
	switch crewKind {
	case CrewKindDiagnoser:
		return DispatchRoleDiagnose
	case CrewKindExtractor:
		return DispatchRoleExtract
	}
	return ""
}

// ServiceName is the Kubernetes Deployment backing a crew kind.
func ServiceName(crewKind string) string {
	switch crewKind {
	case CrewKindDiagnoser:
		return "chora-companion-diagnoser"
	case CrewKindExtractor:
		return "chora-companion-extractor"
	}
	return ""
}
