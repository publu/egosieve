# Annotation guide

This rubric defines labels for a fixed sampled window. Annotators judge only
observable RGB evidence and the stated downstream goal: whether the window is
worth sending to a more expensive manipulation-data pipeline. They do not
infer intent, success, force, depth, or robot safety.

## Readiness decision

- `KEEP`: a meaningful hand–object interaction is visible for enough of the
  window to support downstream labeling or reconstruction, and no visual issue
  makes the evidence unusable.
- `REVIEW`: evidence is ambiguous, borderline, domain-specific, or cannot be
  judged confidently. Review is a first-class answer, not a softer reject.
- `REJECT`: the window has no useful manipulation evidence or an observable
  issue makes it unusable for the declared downstream task.

Annotators first decide which issue labels are known, then make the readiness
decision. A present issue does not mechanically force `REJECT`; for example,
brief occlusion may still leave a useful window.

## Observable issues

- `no_hands`: no human hand is visibly present.
- `low_hand_activity`: hands may be visible, but there is no meaningful hand or
  object-state change.
- `hand_occlusion`: the acting hand or critical contact region is hidden long
  enough to obstruct interpretation.
- `camera_instability`: camera motion or rolling blur prevents stable visual
  tracking; ordinary intentional head motion alone is not an issue.
- `blur`: defocus or motion blur removes task-relevant detail.
- `exposure`: clipping or darkness removes task-relevant detail.
- `scene_cut`: an edit, discontinuity, or decoder jump occurs inside the
  window.
- `duplicate_frames`: frozen or repeated imagery materially removes temporal
  evidence.

An issue is `false` only when the annotator has enough evidence to rule it out.
Otherwise its validity mask is false; missing never means negative.

## Interaction boundaries

The start boundary is the first visible task-relevant hand movement toward an
object or the first contact when approach is outside the window. The end is the
last task-relevant release or stabilization before the hand disengages. Leave a
boundary invalid when it falls outside the window or cannot be localized.
Record timestamps on the source presentation timeline, not frame indices.

## Review protocol

1. Randomize windows within each source stratum and hide model predictions.
2. Collect at least two independent labels for the public test subset.
3. Adjudicate readiness disagreements and boundary differences larger than the
   declared tolerance; retain both original votes.
4. Report class counts, agreement, adjudication rate, and results by capture
   source or embodiment.
5. Audit examples spanning camera placement, lighting, gloves, assistive
   devices, skin tone, and unusual viewpoints without publishing media that
   lacks redistribution consent.

The evaluation manifest records annotator identifiers or pseudonyms, rubric
version, annotation timestamp, and adjudication status. Dataset cards must also
document compensation, consent, privacy review, and media licenses.
