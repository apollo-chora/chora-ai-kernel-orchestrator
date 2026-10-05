package agent

import "google.golang.org/adk/session"

// conditions.go — ADR-197 M-A condition extractor + embedded-default prompt
// version for the Familiar companion.

// FamiliarPromptVersion is the embedded-default prompt version. Familiar has no
// agentconfig YAML (its prompt is composed per-instance from FamiliarConfig);
// this constant is the ADR-197 embedded-default version stamped onto decisions
// until the M-B registry supplies an override.
const FamiliarPromptVersion = "v1"

// InstanceConditions surfaces the reliably-in-state prompt discriminants for the
// Familiar. The per-instance FamiliarConfig (species / growth stage /
// specialization / evolution tier) is resolved via the registry inside the
// dispatch and is not a plain state key; for M-A we surface familiar_id (the
// instance whose config shaped the prompt) + mana_tier. The content_hash on the
// same span captures the full composed per-instance prompt.
func InstanceConditions(state session.ReadonlyState) map[string]string {
	c := map[string]string{}
	if v := condStateString(state, "familiar_id"); v != "" {
		c["familiar_id"] = v
	}
	if v := condStateString(state, "mana_tier"); v != "" {
		c["mana_tier"] = v
	}
	return c
}

func condStateString(state session.ReadonlyState, key string) string {
	if state == nil {
		return ""
	}
	v, err := state.Get(key)
	if err != nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}
