'use client';

import { useState } from 'react';
import { format } from 'date-fns';
import { Edit, Info, Plus, Trash } from 'lucide-react';

import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import Modal from '@/components/Modal';
import { Separator } from '@/components/ui/separator';

import type { PatientMedicationField } from './PatientMedications';
import { capitalize } from '@/lib/utils';

interface PatientMedicationItemProps {
  medication: PatientMedicationField;
  isEditDisabled?: boolean;
  onEdit: () => void;
  isDeleteDisabled?: boolean;
  onDelete: () => void;
  view?: boolean;
}

export default function PatientMedicationItem({
  medication,
  isEditDisabled = false,
  onEdit,
  isDeleteDisabled = false,
  onDelete,
  view = false,
}: PatientMedicationItemProps) {
  const [isDetailsOpen, setIsDetailsOpen] = useState(false);

  return (
    <Card className="col-span-1 rounded-lg transition-all">
      <div className="truncate px-4 pt-6">
        <div className="flex flex-col gap-0.5">
          <h3 className="truncate text-lg font-medium">
            {medication.medication.name}
          </h3>
          <span className="mt-0.5 truncate">
            Dosage: {medication.dosage} {medication.medication.unit}{' '}
            {medication.frequency} (
            {medication.required ? 'Required' : 'Optional'})
          </span>
          <div className="flex items-center gap-1">
            {medication.startDate && (
              <span className="mt-0.5 truncate">
                {format(new Date(medication.startDate), 'dd MMM yyyy')}
              </span>
            )}
            {medication.endDate && (
              <span className="mt-0.5 truncate">
                - {format(new Date(medication.endDate), 'dd MMM yyyy')}
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-2 px-4 py-2 text-xs text-muted-foreground">
        <div className="flex items-center gap-2">
          <Plus className="h-4 w-4" />
          {format(new Date(medication.createdAt), 'dd MMM yyyy')}
        </div>

        <div className="flex flex-wrap gap-2">
          {view ? (
            <Button size="sm" onClick={() => setIsDetailsOpen(true)}>
              <Info className="mr-2 h-4 w-4" /> Details
            </Button>
          ) : (
            <>
              <Button
                size="sm"
                variant="secondary"
                disabled={isEditDisabled}
                onClick={onEdit}
              >
                <Edit className="mr-2 h-4 w-4" /> Edit
              </Button>
              <Button
                size="sm"
                variant="destructive"
                disabled={isDeleteDisabled}
                onClick={onDelete}
              >
                <Trash className="mr-2 h-4 w-4" /> Delete
              </Button>
            </>
          )}
        </div>
      </div>

      <Modal
        isOpen={isDetailsOpen}
        onClose={() => setIsDetailsOpen(false)}
        className="h-fit md:overflow-y-auto"
      >
        <div className="space-y-6 px-4 py-2 text-foreground">
          <div className="w-full space-y-2">
            <div>
              <h3 className="text-lg font-medium">Medication Details</h3>
              <p className="text-sm text-muted-foreground">
                Details of the medication.
              </p>
            </div>
            <Separator className="bg-primary/10" />
          </div>
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-1">
              <p className="font-medium text-primary">Medication</p>
              <p>
                {medication.medication.name}
                {medication.required && <span> (Required)</span>}
              </p>
            </div>

            {medication.dosage && (
              <div className="flex flex-col gap-1">
                <p className="font-medium text-primary">Dosage</p>
                <p>
                  {medication.dosage} {medication.medication.unit}
                </p>
              </div>
            )}

            {medication.frequency && (
              <div className="flex flex-col gap-1">
                <p className="font-medium text-primary">Frequency</p>
                <p>{capitalize(medication.frequency)}</p>
              </div>
            )}

            {medication.startDate && (
              <div className="flex flex-col gap-1">
                <p className="font-medium text-primary">
                  Medication Start Date
                </p>
                <p>{format(new Date(medication.startDate), 'dd MMM yyyy')}</p>
              </div>
            )}

            {medication.endDate && (
              <div className="flex flex-col gap-1">
                <p className="font-medium text-primary">Medication End Date</p>
                <p>{format(new Date(medication.endDate), 'dd MMM yyyy')}</p>
              </div>
            )}

            {medication.medication.description && (
              <div className="flex flex-col gap-1">
                <p className="font-medium text-primary">
                  Medication Description
                </p>
                <p
                  dangerouslySetInnerHTML={{
                    __html: medication.medication.description.replace(
                      /\n/g,
                      '<br />'
                    ),
                  }}
                />
              </div>
            )}

            {medication.notes && (
              <div className="flex flex-col gap-1">
                <p className="font-medium text-primary">Additional Notes</p>
                <p
                  dangerouslySetInnerHTML={{
                    __html: medication.notes.replace(/\n/g, '<br />'),
                  }}
                />
              </div>
            )}
          </div>
        </div>
      </Modal>
    </Card>
  );
}